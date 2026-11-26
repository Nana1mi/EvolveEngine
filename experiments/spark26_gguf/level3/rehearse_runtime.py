"""Real-Pi Level 3 integration with local scripted SSE and a synthetic Worker only."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

from pi_runtime import ROOT, PI, cfg, profile_and_runtime
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_guest import sandbox
from evolveengine.v3_level3_research import PublicCorpus
from evolveengine.v3_level3_workload import FIRST_LIVE_PAIR
from evolveengine.v3_level3_run import CORE, EMPTY, SCENES, WorkerResult, explore
from evolveengine.v3_level3_source import SourceContract

PY = 'python/sglang/srt/managers/scheduler.py'
CU = 'python/sglang/kernels/aot/csrc/level3_fixture.cu'
CPP = 'python/sglang/kernels/aot/csrc/level3_fixture.cpp'
TRITON = 'python/sglang/kernels/jit/level3_fixture.triton'
EXT = 'extensions/marker.mjs'
DIAG = 'diagnostics/marker.py'
DIAG_SOURCE = 'print(42)\n'
MARKER = 'level3-fixture-H1-actually-ran'
EXT_SRC = '''export default function(pi){pi.registerTool({name:'l3_marker', label:'Fixture marker',
    description:'Confirm the submitted Harness is loaded', parameters:{type:'object',properties:{}},
    async execute(){return {content:[{type:'text',text:'level3-fixture-H1-actually-ran'}]};}});}'''
SYSTEM = 'Execute the scripted local Level 3 integration. All performance and worker evidence are synthetic.'


class FixtureWorker:
    def __init__(self, root, contract, memoization=None):
        self.root, self.contract = root, contract
        self.memoization = memoization

    def __call__(self, operation, candidate, build, cap):
        if candidate.edits:
            files = dict(candidate.edits)
            assert files[PY].endswith(b'# level3-fixture\n')
            assert len(files[CU]) > 192 * 1024 and b'unfinished_wip' not in files[CU]
            assert files[CPP] == b'int fixture() { return 0; }\n'
            assert files[TRITON] == b'# triton fixture\n'
            assert candidate.build_args['jobs'] == 2
        status = {'develop_build':'built', 'develop_quality':'valid', 'develop_measure':'observed'}[operation]
        data = dict(synthetic=True)
        if operation == 'develop_measure':
            data['scenes'] = {name:dict(dev_score=103 if candidate.edits else 100) for name in SCENES}
        return self._receipt(operation,candidate,build,status,data)

    def measure(self, candidate, build, cap, pair_index):
        # The first candidate supplies a memorized/wrong response, despite a valid
        # separate quality receipt and high claimed timing score.
        result = self('develop_measure', candidate, build, cap)
        stale = bool(candidate.edits) and pair_index == FIRST_LIVE_PAIR
        input_pin = digest(canonical(['synthetic pair', pair_index]))
        output_pin = digest(b'memoized' if stale else b'matched greedy output')
        if self.memoization and pair_index == FIRST_LIVE_PAIR:
            input_pin = self.memoization['input_sha256']
            output_pin = self.memoization['stale_sha256' if stale else 'actual_sha256']
        data = dict(result.data, measurement_pair=pair_index,
                    measurement_inputs_sha256=input_pin, timed_outputs_sha256=output_pin)
        return self._receipt('develop_measure', candidate, build, 'observed', data)

    def diagnose(self, candidate, build, source, cap):
        assert candidate.edits and build.status == 'built' and self.verify(build)
        assert source == DIAG_SOURCE and cap <= 180000
        data = dict(non_scoring=True, synthetic=True, script_sha256=digest(source.encode()))
        return self._receipt('develop_diagnostic',candidate,build,'completed_non_scoring',data)

    def _receipt(self, operation, candidate, build, status, data):
        payload = dict(operation=operation, engine=candidate.digest, status=status, node_ms=17,
                       build=build.evidence if build else None, data=data)
        raw = canonical(payload)
        address = digest(raw)
        self.root.mkdir(exist_ok=True)
        path = self.root/address
        if not path.exists():
            path.write_bytes(raw)
        return WorkerResult(**payload, evidence=address)

    def verify(self, result):
        payload = dict(operation=result.operation, engine=result.engine, status=result.status,
                       node_ms=result.node_ms, build=result.build, data=result.data)
        raw = (self.root/result.evidence).read_bytes()
        return sha256(raw).hexdigest() == result.evidence and raw == canonical(payload)


def run(output, contract, counter, scenario, research=None, memoization=None, objective=False):
    if objective:
        config, limits = cfg.validate_objective_config(json.loads(
            (Path(__file__).parent/'objective-exploration.json').read_bytes()))
    else:
        config, limits = cfg.load_config(cfg.CONFIG)
    tools = None
    resources = config.get('guest_resources') if objective else None
    profile, runtime = profile_and_runtime(limits, SYSTEM, tools, resources=resources)
    cost = cfg.UsageLedger() if objective else cfg.ApiCostBudget(cfg.cost_cap(config['budget']['api_cost_rmb']))
    errors, observed, refs = [], [], {}

    def rpc(operation, **args):
        return ('ee_rpc', dict(operation=operation, arguments=args))

    def bash(command):
        return ('bash', dict(command=command))

    def result(request):
        msg = request['messages'][-1]
        assert msg['role'] == 'tool'
        row = json.loads(msg['content'])
        assert row['status'] == 200, row
        return row['result']

    prelude = []
    if research is not None:
        assert scenario == 'delivery'
        prelude.append(('facts',{},None))
        for kind,query in (('web','CUDA C++ best practices'),('paper','FlashAttention-3 Asynchrony'),('source','CUTLASS MMA')):
            hits = research.search(query)
            ref = next(row for row in hits['results'] if row['kind']==kind)
            prelude.append(('research_search',dict(query=query),hits))
            page = research.read(ref['sha256'])
            prelude.append(('research_read',dict(identity=ref['sha256']),page))
            if kind=='paper':
                assert page['truncated'] is True
                offset = page['next_offset']
                prelude.append(('research_read',dict(identity=ref['sha256'],offset=offset),research.read(ref['sha256'],offset)))

    def choose(segment, turn, request):
        if turn == 0:
            assert [m['role'] for m in request['messages']] == ['system', 'user']
        else:
            assistants = [m for m in request['messages'] if m['role'] == 'assistant']
            assert assistants[-1]['reasoning_content'] == 'synthetic reasoning replay'
        if scenario == 'second-truncation':
            assert turn == 0 and 1 <= segment <= (3 if objective else 2)
            return 'length'
        if segment != 1:
            assert any(tool['function']['name'] == 'l3_marker' for tool in request['tools'])
            if turn == 0:
                return 'l3_marker', {}
            if turn == 1:
                assert MARKER in request['messages'][-1]['content']
                return bash(f'grep -q unfinished_wip {CU} && test ! -e untracked.txt && test ! -e truncation-executed && printf fresh-workspace')
            assert turn == 2 and 'fresh-workspace' in request['messages'][-1]['content']
            return 'length' if segment == 2 else rpc('finish_exploration', engine=refs['candidate'])
        if prelude and turn<=len(prelude):
            if turn:
                previous = prelude[turn-1][2]
                response = result(request)
                if previous is None:
                    assert response['research_scope']=='reviewed_public_cache'
                else:
                    assert response==previous
            if turn<len(prelude):
                op,args,_ = prelude[turn]
                return rpc(op,**args)
        turn -= len(prelude)
        if turn == 0:
            return bash(f"test -f {PY} && printf '\\n# level3-fixture\\n' >> {PY} && "
                        f"mkdir -p $(dirname {CU}) $(dirname {TRITON}) /scratch/harness/extensions /scratch/harness/diagnostics && "
                        f"printf 'print(42)\\n' > /scratch/harness/{DIAG} && "
                        f"printf 'int fixture() {{ return 0; }}\\n' > {CPP} && "
                        f"printf '# triton fixture\\n' > {TRITON} && "
                        f"printf '// ' > {CU} && head -c 221000 /dev/zero | tr '\\000' x >> {CU}")
        if turn == 1:
            return rpc('candidate_submit', paths=[PY,CU,CPP,TRITON], build_args={'jobs':2})
        if turn == 2:
            refs['candidate'] = result(request)['engine']
            return rpc('develop_build', engine=refs['candidate'])
        if turn == 3:
            assert result(request)['status'] == 'built'
            return rpc('diagnostic_submit', engine=refs['candidate'], path=DIAG)
        if turn == 4:
            response = result(request)
            assert response['status'] == 'completed_non_scoring' and response['data']['non_scoring']
            return rpc('develop_quality', engine=refs['candidate'])
        if turn == 5:
            assert result(request)['status'] == 'valid'
            return rpc('develop_measure', engine=refs['candidate'])
        if turn == 6:
            response = result(request)
            assert response['status'] == 'timed_output_mismatch' and 'performance' not in response
            return rpc('develop_measure', engine=refs['candidate'])
        if turn > 6:
            turn -= 1
        if turn == 6:
            response = result(request)
            assert response['status'] == 'observed' and response['performance']['equal_scene_geomean'] == 1.03
            return 'write', dict(path=f'/scratch/harness/{EXT}', content=EXT_SRC)
        if turn == 7:
            return rpc('harness_submit', paths=[EXT], load=dict(EMPTY, extensions=[EXT]))
        if turn == 8:
            refs['h1'] = result(request)['version']
            refs['h0'] = (output/'harness-lineage/active').read_text().strip()
            return rpc('harness_activate', version=refs['h1'], expected=refs['h0'])
        if turn == 9:
            result(request)
            return bash(f"printf '\\n// unfinished_wip\\n' >> {CU}; touch untracked.txt")
        if turn == 10:
            return rpc('candidate_submit', paths=[CU], parent=refs['candidate'])
        if turn == 11:
            refs['wip'] = result(request)['engine']
            return rpc('save_checkpoint', engine=refs['candidate'], wip=refs['wip'])
        assert turn == 12
        return rpc('continue_session', checkpoint=result(request)['checkpoint'])

    def lease_factory(active_limits, segment, fragments):
        turn = 0

        @contextmanager
        def send(payload, _timeout):
            nonlocal turn
            request = json.loads(payload)
            try:
                assert request['max_tokens'] == limits.per_turn_output
                assert request['reasoning_effort'] == 'max' and request['thinking'] == {'type':'enabled'}
                step = choose(segment, turn, request)
            except (AssertionError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'segment {segment}, turn {turn}: {type(exc).__name__}: {exc}')
                raise ValueError('offline model fixture rejected request') from exc
            observed.append(dict(segment=segment, turn=turn, tool=step if isinstance(step,str) else step[0]))
            truncated = step == 'length'
            tool, args = ('bash', {}) if truncated else step
            args = '{"command":"touch truncation-executed' if truncated else canonical(args).decode()
            def event(delta=None, finish=None, usage=None):
                row = dict(id=f'l3-fixture-{segment}-{turn}', model='deepseek-flash',
                           system_fingerprint=cfg.narrow.FINGERPRINT,
                           choices=[dict(index=0,delta=delta or {},finish_reason=finish)], usage=usage)
                return b'data: '+canonical(row)+b'\n\n'
            n = request['max_tokens'] if truncated else 21
            tokens = dict(prompt_tokens=250, completion_tokens=n, total_tokens=250+n,
                          completion_tokens_details=dict(reasoning_tokens=n if truncated else 5),
                          prompt_cache_hit_tokens=100,prompt_cache_miss_tokens=150)
            parts = [event(dict(role='assistant',reasoning_content='synthetic reasoning replay')),
                     event(dict(tool_calls=[dict(index=0,id=f'call_{segment}_{turn}',type='function',
                                                 function=dict(name=tool,arguments=args))])),
                     event(finish='length' if truncated else 'tool_calls',usage=tokens),b'data: [DONE]\n\n']
            turn += 1
            yield parts

        lease = cfg.ExplorationLease(active_limits, send, counter, config['model']['context_window'],
                                    output/f'segment-{segment}-model-attempts', api_cost=cost,
                                    model_socket_idle_seconds=config['transport']['relay_socket_idle_seconds'],
                                    expected_prompt_fragments=fragments)
        if objective:
            lease.require_full_generation_window = False
            lease.minimum_generation = config['model']['minimum_generation_tokens']
            lease.context_margin = config['model']['context_margin_tokens']
        return lease

    def launch(engine, snapshot, prompt, gateway, *, seconds, draft):
        return sandbox(PI, contract, engine, snapshot, runtime, profile, SYSTEM, prompt, gateway,
                       seconds=seconds, tools=tools, draft=draft, resources=resources)

    worker = FixtureWorker(output/'synthetic-worker', contract, memoization)
    r = explore(output, contract, worker, launch, lease_factory,
                runtime, 'Exercise the registered offline tool flow.', limits,
                node_ms=None if objective else 60_000, calls=None if objective else 12, reservations=dict.fromkeys(CORE|{'develop_diagnostic'},5_000),
                research=research, bootstrap_measure=True, paired_timing=True)
    (output/'rehearsal.json').write_bytes(canonical(dict(synthetic=True, observed=observed, errors=errors,
                                                        api_cost_budget=cost.snapshot(),
                                                        research_corpus_sha256=research.identity if research else None,
                                                        checked_research_tools=[op for op,_,_ in prelude],
                                                        memoization_witness_sha256=digest(canonical(memoization)+b'\n')
                                                        if memoization else None))+b'\n')
    assert not errors, errors
    assert all(s['report']['isolated_state_removed'] and s['report']['socket_removed']
               and s['report']['unchanged_install'] for s in r['segments']), r['stop_cause']
    if scenario == 'delivery':
        assert r['status'] == 'delivered' and len(r['segments']) == 3, r['stop_cause']
        assert r['final']['chosen']['engine'] == refs['candidate'] and r['final']['active_harness'] == refs['h1']
        assert r['development_reference'] == r['jobs'][2]['evidence']
        assert [row['operation'] for row in r['jobs']] == [
            'develop_build','develop_quality','develop_measure',
            'develop_build','develop_diagnostic','develop_quality',
            'develop_measure','develop_measure','develop_measure','develop_measure',
        ]
        measurements = [row for row in r['jobs'] if row['operation'] == 'develop_measure']
        assert [row['measurement_pair'] for row in measurements] == [0, FIRST_LIVE_PAIR, FIRST_LIVE_PAIR,
                                                                      FIRST_LIVE_PAIR + 1, FIRST_LIVE_PAIR + 1]
        assert measurements[2]['scoring_status'] == 'timed_output_mismatch' and 'performance' not in measurements[2]
        assert measurements[4]['performance']['reference'] == measurements[3]['evidence']
        for s in r['segments'][1:]:
            assert s['entry']['harness'] == refs['h1'] and s['report']['wip'] == refs['wip']
            assert any(c['name'] == 'l3_marker' and not c['error'] for c in s['report']['pi_tool_executions'])
        assert len(r['segments'][1]['report']['pi_tool_executions']) == 2
    else:
        expected = ('repeated_failure_paused', 3) if objective else ('recovery_cap_reached', 2)
        assert (r['stop_cause'], len(r['segments'])) == expected, r['stop_cause']
        assert not r['final'] and not any(s['report']['pi_tool_executions'] for s in r['segments'])
    assert r['wallet']['requests'] == len(observed)
    assert r['wallet']['input'] == len(observed)*250
    if objective:
        assert cost.snapshot()['unresolved_requests'] == 0
    else:
        assert cost.reserved == 0
    print(json.dumps(dict(synthetic=True, scenario=scenario, status=r['status'], stop=r['stop_cause'],
                          segments=len(r['segments']), calls=len(observed), worker_operations=len(r['jobs']),
                          output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new local evidence directory')
    parser.add_argument('--objective', action='store_true', help='Exercise the engineering metering and recovery path')
    parser.add_argument('--archive', type=Path, help='use the real pinned E0 tree instead of the small local fixture')
    parser.add_argument('--scenario', choices=('delivery','second-truncation'), default='delivery')
    parser.add_argument('--public-corpus', type=Path)
    parser.add_argument('--public-corpus-sha256')
    parser.add_argument('--memoization-witness', type=Path, help='Replay output fingerprints from prior hardware observations')
    parser.add_argument('--memoization-witness-sha256')
    args = parser.parse_args()
    if bool(args.public_corpus)!=bool(args.public_corpus_sha256) or (args.public_corpus and args.scenario!='delivery'):
        parser.error('research requires a pinned corpus and the delivery scenario')
    research = PublicCorpus(args.public_corpus,args.public_corpus_sha256,e0_archive=args.archive) if args.public_corpus else None
    if bool(args.memoization_witness)!=bool(args.memoization_witness_sha256) or (args.memoization_witness and args.scenario!='delivery'):
        parser.error('memoization replay requires a pinned witness and the delivery scenario')
    memoization = None
    if args.memoization_witness:
        with args.memoization_witness.open('rb') as f:
            raw = f.read(4097)
        if len(raw) > 4096 or digest(raw) != args.memoization_witness_sha256:
            parser.error('memoization witness size or fingerprint mismatch')
        memoization = json.loads(raw)
        if (raw != canonical(memoization)+b'\n'
                or memoization['kind'] != 'offline_replay_of_hardware_output_fingerprints'
                or memoization['actual_sha256'] == memoization['stale_sha256']):
            parser.error('memoization witness requires a verified stale output difference')
    contract = (SourceContract.acquire_spark26_e0(args.archive, ROOT) if args.archive
                else SourceContract({PY:b'# local runtime fixture\n'}))
    tokenizer = Path(os.environ['LOCALAPPDATA'])/'EvolveEngine/deepseek-flash-1017-08c8657f-offline'
    run(args.output, contract, cfg.narrow.PublicPromptCounter(tokenizer), args.scenario,
        research=research, memoization=memoization, objective=args.objective)


if __name__ == '__main__':
    main()
