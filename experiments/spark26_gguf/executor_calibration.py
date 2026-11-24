"""Nonblind one-file Pi budget calibration. Real runs require E0 and candidate raw evidence."""
import argparse
from difflib import unified_diff
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolveengine.deepseek import Provider, ProviderBudget
from evolveengine.pi import PiAgent
from evolveengine.schema import Cost
import reference_pair_audit
import reference_qkv as qkv
import reference_qkv_quality_audit as quality
import reference_shadow_audit as shadow

VERSION = '1.5.2'
CAP = Cost(wall_ms=300_000, input_tokens=40_000, output_tokens=8_000)
MAX_TURNS = 6
MAX_TOOLS = 10
MAX_LINES = 128
BUNDLE_SHA = 'dadf668859f15b71e87d20bff01ccb6ae1850b38595d2fc71f5c867e181479f9'
GUARD = Path(__file__).with_name('executor_calibration_guard.ts').resolve()
SRC = qkv.RELATIVE.as_posix()


class CalibrationBudget(ProviderBudget):
    def turn_output_ceiling(self, input_bound):
        if len(self.rows) >= MAX_TURNS:
            raise ValueError('Provider turn cap exhausted')
        if input_bound > self.limit.input_tokens-self.spent.input_tokens:
            raise ValueError('Provider input reservation does not fit')
        return min(super().turn_output_ceiling(input_bound), self.limit.output_tokens-self.spent.output_tokens)


def trusted_evidence(service_dir, e0_quality_dir, quality_dir, residence_dir, e0_residence_log, e0_calibration_dir):
    sr = reference_pair_audit.audit(service_dir, BUNDLE_SHA)
    if not sr['outcome']['engineering_screen_passed']:
        raise ValueError('Service screen did not pass')
    lock = json.loads(quality.LOCK.read_bytes())
    q = quality.compare(e0_quality_dir, quality_dir, lock)
    r = shadow.audit(residence_dir, e0_residence_log, e0_calibration_dir, ('QKVParallelLinear',))
    return dict(candidate_sha256=q['candidate_sha256'], service_bundle=sr['bundle_sha256'],
                deterministic_manifest=q['raw_evidence_manifest'], shadow_shapes=r['exact_shadow_checks'])


def prompt_for(old, new):
    offset = old.decode().splitlines().index('    def apply(') + 1
    diff = ''.join(unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
                                'a/'+SRC, 'b/'+SRC))
    return (f'This is a nonblind patch-replay calibration. Apply only the supplied unified diff, '
            f'exactly, leaving all other bytes alone. Only read/edit {SRC}. '
            f'The anchor is GGUFLinearMethod.apply near line {offset}; any read must specify a 1-based offset '
            f'and an explicit limit of at most {MAX_LINES} lines. Use an edit tool for the actual change. '
            'Do not modify any other file or claim success without an edit. '
            'Finish with exactly one JSON object {"applied":true} if applied, else {"applied":false}.\n\n'+diff)


def check_checkout(root, base, wanted):
    target = root/qkv.RELATIVE
    if target.is_symlink() or target.resolve() != target:
        raise ValueError('Target escaped or was replaced by a link')
    if {entry.relative_to(root) for entry in root.rglob('*') if not entry.is_dir() or entry.is_symlink()} != {qkv.RELATIVE}:
        raise ValueError('Unexpected files in one-file checkout')
    data = target.read_bytes()
    if data == base:
        raise ValueError('Empty patch')
    if sha256(data).hexdigest() != wanted:
        raise ValueError('Nonempty diff did not exactly match the accepted reference; cannot bind GPU evidence')
    compile(data, SRC, 'exec')
    return data


def run_calibration(agent, src, evidence, result_root):
    base = src.read_bytes()
    reference = qkv.edited(base)
    want = evidence['candidate_sha256']
    if sha256(reference).hexdigest() != want:
        raise ValueError('Reference parity does not match the independently accepted candidate')
    result_root.mkdir(exist_ok=False)
    tooling = result_root/'tool-receipts.jsonl'
    receipt = dict(scope='nonblind supplied reference; not a production or formal blind trial',
                   version=VERSION, limits={**CAP.record(), 'turns':MAX_TURNS, 'tools':MAX_TOOLS,
                                            'max_read_lines':MAX_LINES}, external_evidence=evidence,
                   status='infra_or_edit_failure')
    try:
        receipt['identity'] = agent.preflight()
        with tempfile.TemporaryDirectory(prefix='ee-gguf-calibration-') as th:
            root = Path(th).resolve()
            target = root/qkv.RELATIVE
            target.parent.mkdir(parents=True)
            target.write_bytes(base)
            start = time.monotonic()
            with CalibrationBudget(agent.provider, CAP) as broker:
                try:
                    answer, _ = agent.session(root, prompt_for(base, reference), broker, editing=(SRC,), read_scope=(SRC,),
                                             confined_guard=(GUARD, dict(receipt=str(tooling),max_tools=MAX_TOOLS,max_lines=MAX_LINES)))
                finally:
                    settled = broker.settled.is_set()
                    billed = broker.spent if settled else CAP
                    receipt['provider'] = dict(turns=broker.rows, settled=settled, usage=billed.record())
                    receipt['edit_wall_ms'] = round(1000*(time.monotonic()-start))
            tools = [json.loads(line) for line in tooling.read_text(encoding='utf-8').splitlines()]
            if (not settled or not 1 <= len(broker.rows) <= MAX_TURNS or any('usage' not in row for row in broker.rows)
                or not 1 <= len(tools) <= MAX_TOOLS or not any(row['tool']=='edit' for row in tools)
                or not all(row['ordinal']==i and row['accepted'] is True and row['tool'] in {'read','edit'}
                           for i,row in enumerate(tools,1))
                or receipt['edit_wall_ms'] > CAP.wall_ms or billed.input_tokens > CAP.input_tokens
                or billed.output_tokens > CAP.output_tokens):
                raise ValueError('Calibration has incomplete receipts or exceeded its lease')
            if answer != {'applied': True}:
                raise ValueError('Pi did not report successful application')
            started_check = time.monotonic()
            result = check_checkout(root,base,want)
            receipt['external_syntax_check_ms'] = round(1000*(time.monotonic()-started_check),3)
            receipt['result_sha256'] = sha256(result).hexdigest()
            patch_lines = unified_diff(base.decode().splitlines(keepends=True), result.decode().splitlines(keepends=True),
                                       'a/'+SRC,'b/'+SRC)
            (result_root/'candidate.diff').write_bytes(''.join(patch_lines).encode())
            receipt['status'] = 'pass'
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        receipt['failure_class'] = type(error).__name__
        # ProviderHTTPError exposes the status but response bodies, prompts and credentials are not printed.
        receipt['reason'] = str(error)[:350]
    finally:
        (result_root/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','pi_cli','output','service','e0_quality','quality','residence','e0_residence_log','e0_calibration'):
        parser.add_argument(name, type=Path)
    args=parser.parse_args()
    ev=trusted_evidence(args.service,args.e0_quality,args.quality,args.residence,args.e0_residence_log,args.e0_calibration)
    agent=PiAgent(['node',str(args.pi_cli.resolve(strict=True))],VERSION,
                  Provider('https://api.deepseek.com','deepseek-flash','DEEPSEEK_API_KEY','DeepSeek-V4.1-Flash',2500))
    result=run_calibration(agent,args.source,ev,args.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in {'provider','external_evidence'}},indent=2))
    raise SystemExit(result['status']!='pass')


if __name__=='__main__':
    main()
