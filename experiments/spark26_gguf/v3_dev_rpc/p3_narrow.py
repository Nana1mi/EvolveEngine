"""One bounded SGLang scope-validation session on the registered worker."""
import argparse
import ast
from dataclasses import asdict, replace
import os
from pathlib import Path
import re
import sys
from time import monotonic

import p3
import node_client
from evolveengine.v3_context_control import ARTIFACTS, MODEL_REVISION, TOKENIZERS_VERSION, PublicPromptCounter
from evolveengine.v3_provider import GENERATION_TRUNCATED, MaxLease, LeaseError

RUN = 'v3-p3-narrow-20261124-r01'
FINGERPRINT = 'bc19ba97cbe016479903222bb1578db6'
TARGET = 'GGUFLinearMethod.apply'
FILE = p3.ALLOWED[0]
TOOLS = None
SYSTEM = '''You are running one bounded development experiment on the isolated SGLang/CUDA GGUF Q8_0 Engine. Preserve the weight format, numerical behavior, vocabulary, EOS and request isolation. Seek a useful improvement on the assigned code path. The only Engine edits accepted by the trusted gateway are inside the body of GGUFLinearMethod.apply in the named file. Do not edit its signature, class, surrounding code, or test infrastructure. You may freely choose the approach and use temporary scripts.

The source is in the working directory. The read tool returns a UTF-8 page with a byte offset; use the task's offset to see the target. The trusted ee_rpc tool takes operation and arguments. Use develop_quality with {paths:[relative Engine path]} for the current changed candidate. After a valid quality receipt, deliver_engine with the same paths explicitly signs that exact Engine. Optional develop_measure uses the same paths and returns four directional metrics from one same-worker E0 and one candidate boot. A valid candidate can be delivered without measuring. Harness updates are optional via harness_status, harness_submit and harness_activate. The outside controller records exact artifacts and costs. Report what happened, including failed checks or a decision to leave E0 unchanged.'''


class NarrowContract(p3.SourceContract):
    def __init__(self, source):
        super().__init__({FILE: source}, [FILE], max_files=1, max_bytes=160*1024,
                         max_text_growth=1024, max_new_literal=1024)
        self.source = source
        self.first, self.last, self.outer_ast = self.project(source)
        self.lines = source.splitlines(keepends=True)

    @staticmethod
    def project(raw):
        tree = ast.parse(raw.decode('utf-8'))
        classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'GGUFLinearMethod']
        p3.need(len(classes) == 1, 'target class must be unique')
        methods = [x for x in classes[0].body if isinstance(x, ast.FunctionDef) and x.name == 'apply']
        p3.need(len(methods) == 1, 'target function must be unique')
        method = methods[0]
        first = min([method.body[0].lineno, *(x.lineno for x in getattr(method.body[0], 'decorator_list', []))])-1
        last = method.end_lineno
        method.body = []
        return first, last, ast.dump(tree, include_attributes=False)

    def from_args(self, args):
        candidate = super().from_args(args)
        p3.need(len(candidate.edits) == 1, 'submit a changed candidate in the target function')
        raw = dict(candidate.edits)[FILE]
        try:
            first, last, outer_ast = self.project(raw)
            lines = raw.splitlines(keepends=True)
        except (SyntaxError, UnicodeError, RecursionError) as exc:
            raise ValueError('invalid target source') from exc
        # Prefix/suffix are byte-identical. AST comparison rejects escaping the body via indentation.
        p3.need(first == self.first and lines[:first] == self.lines[:self.first] and
                lines[last:] == self.lines[self.last:] and outer_ast == self.outer_ast,
                'edits outside the nominated function body')
        return candidate

    def prompt(self):
        offset = sum(len(line) for line in self.lines[:self.first-6])
        return (f'Improve the performance of the sharded GGUF linear path inside {TARGET} in {FILE}, '
                'while preserving behavior. Only its body may change. '
                f'Read near byte offset {offset} (body lines {self.first+1}–{self.last}). '
                'Choose the implementation yourself. Submit a changed candidate for develop_quality and explicitly deliver '
                'it if you accept the result. Use develop_measure only if useful. Summarize the evidence and stop.')


class NarrowLease(MaxLease):
    """Public-template count plus preregistered margin, reconciled against reported usage."""
    def __init__(self, limits, transport, counter, *, expected_prompt_fragments=()):
        self.counter = counter
        self.assistant_hashes = []
        super().__init__(limits, transport, fingerprint=FINGERPRINT, expected_prompt_fragments=expected_prompt_fragments)

    def _input_ceiling(self, req, tools, meta):
        try:
            parts = self.counter.reservation(req)
        except (AssertionError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise LeaseError('input reservation unavailable') from exc
        meta['input_reservation'] = parts
        return parts['local_candidate_bound']

    def _record_reasoning(self, ids, reason):
        super()._record_reasoning(ids, reason)
        self.assistant_hashes.append(reason)

    def _replay_matches(self, messages, prior):
        assistants = [m for m in messages if m['role'] == 'assistant']
        return (all(type(m.get('reasoning_content')) is str for m in assistants) and
                super()._replay_matches(messages, prior) and
                [p3.digest(m['reasoning_content'].encode()) for m in assistants] == self.assistant_hashes)


def prereq(source, archive):
    p3.need(p3.digest(source) == p3.PLAN['gguf_py'], 'pinned E0 source SHA mismatch')
    accepted = p3.record(p3.HERE/'g1a-03.json')
    p3.need(accepted['status'] == 'pass_g1a_pre_pilot_scope' and
            p3.bundle_file('bundle.sha256') == p3.bundle_manifest() and
            p3.digest(p3.bundle_manifest()) == accepted['bundle_sha'], 'registered worker bundle drift')
    offline = p3.record(p3.HERE/'p3_context_2x2/offline-rehearsal.json')
    paid = p3.record(p3.HERE/'p3_context_2x2/paid-receipt-20261123.json')
    p3.need(paid['manifest_sha256'] == offline['manifest_sha256'] and len(paid['rows']) == 4 and
            all(r['fingerprint_matches'] and r['status'] == 'settled' for r in paid['rows']),
            'missing pinned diagnostic evidence')
    for row in list(accepted['cases'].values()) + accepted['common_runner']['worker']:
        saved = archive/Path(row['archive']).name
        p3.need(p3.audit(saved, archive/'spark26-v32-g1-d3.lv2uVT', archive/'g1-dev-fJPjW8') == row['result'],
                'registered G1A evidence does not replay')
    p3.PI.verify()


def make_worker(ssh, archive):
    def run_job(op, edits, engine, base, lease, job):
        job.update(operation=op, engine=engine, source_sha=p3.digest(edits.get(FILE, base[FILE])), lease=lease)
        path, rc = p3.invoke(ssh, op, edits, engine, base, lease)
        job.update(archive=path.name, worker_exit=rc)
        if rc == 7:
            r = p3.failed_candidate(path, op, engine, lease)
        else:
            p3.need(rc == 0, 'GPU worker did not provide a complete archive')
            r = p3.audit(path, archive/'spark26-v32-g1-d3.lv2uVT', archive/'g1-dev-fJPjW8')
        patch = p3.record(path/'service/patch.json')
        p3.need(patch['engine'] == engine and patch['operation'] == op and
                p3.digest((path/'service/candidate-gguf.py').read_bytes()) == job['source_sha'] and
                type(r['node_ms']) is int and 0 < r['node_ms'] <= lease, 'trusted receipt source or lease mismatch')
        public = {k:r[k] for k in ('scenes','max_tv','eos_passed','mixed_passed','residency','failure') if k in r}
        receipt = p3.DevReceipt(engine, op, r['status'], r['node_ms'], r['raw_sha'], public)
        job.update(status=receipt.status, node_ms=receipt.node_ms, receipt_id=receipt.receipt_id, summary=public)
        return receipt
    return run_job


def run(source, output, counter, transport, worker, *, run_id=RUN, limits=p3.CAP,
        gpu_node_ms=p3.NODE, development_calls=p3.DEV_CALLS, launcher=Path(__file__),
        lease_factory=NarrowLease, stop_on_delivery=False, registered_config=None, truncation_recoveries=0,
        operation_ms=p3.UNIT):
    began = monotonic()
    narrow = NarrowContract(source)
    output.mkdir(exist_ok=False)
    profile = dict(p3.pilot_profile(), maxTokens=limits.per_turn_output)
    if registered_config is not None:
        profile['contextWindow'] = registered_config['model']['context_window']
    system = SYSTEM
    if stop_on_delivery:
        system += '\n\nThe controller stops after your first accepted deliver_engine. Measure and record your chosen candidate before delivering if you have evidence of a performance improvement.'
    elif registered_config is not None:
        system += ('\n\nExplore and measure further variants when useful. You can sign a qualified candidate as a '
                   'checkpoint and continue working. When you decide to finish, explicitly sign the exact candidate '
                   'you choose, then summarize your measurements and tradeoffs. Each accepted delivery replaces '
                   'the previous selection in the experiment record.')
    runtime = p3.runtime_for(p3.PI, (p3.SMOKE/'package-lock.json').read_bytes(), profile, system, TOOLS, paged_read=True)
    base = {FILE: source}
    snapshot = p3.Snapshot.make({}, p3.EMPTY, runtime)
    lineage = p3.Lineage(output/'harness-lineage', runtime)
    initial = lineage.initialize(snapshot)
    prompt = narrow.prompt()
    if registered_config is not None and not stop_on_delivery:
        prompt = prompt.removesuffix(' Summarize the evidence and stop.')
    config = dict(run=run_id, launcher_sha=p3.digest(launcher.read_bytes()), engine=p3.source_identity(base),
                  start_H=initial, runtime=runtime, target=TARGET, file=FILE, body_lines=[narrow.first+1,narrow.last],
                  prompt_sha=p3.digest(prompt.encode()), system_sha=p3.digest(system.encode()),
                  fingerprint=FINGERPRINT, profile=profile, limits=asdict(limits), gpu_node_ms=gpu_node_ms,
                  development_calls=development_calls, job_reservation_ms=operation_ms, asset_revision=MODEL_REVISION,
                  assets=dict(ARTIFACTS), tokenizers=TOKENIZERS_VERSION, single_episode=True)
    if registered_config is not None:
        config['exploration_config'] = registered_config
        config['driver_sha'] = p3.digest(Path(__file__).read_bytes())
    p3.write_once(output/'registration.json', config)
    baseline, jobs = {}, []
    report = dict(run=run_id, status='incomplete', stage='same_worker_E0', baseline=baseline, workers=jobs,
                  requested_scope='single_engine_function', start_H=initial)
    model = policy = cal = None
    models = []
    checkpoints = {}
    if registered_config is not None:
        report['segments'] = []

    def cumulative_usage():
        return dict(input=sum(m.spent_input for m in models),
                    output_including_reasoning=sum(m.spent_output for m in models))

    try:
        e0 = p3.SourceContract.from_args(narrow, {'files': {}})
        if registered_config is not None:
            baseline.update(operation='develop_measure', engine=e0.digest, reservation_ms=operation_ms)
            p3.write_once(output/'E0-start.json', baseline)
        cal = worker('develop_measure', {}, e0.digest, base, operation_ms, baseline)
        p3.need(cal.status == 'observed', 'same-worker E0 failed')
        p3.metrics(cal.data['scenes'])
        p3.write_once(output/'E0.json', baseline)
        print(f'E0 verified on registered worker ({cal.node_ms/1000:.3f} node seconds)', flush=True)
        remaining = round(limits.wall_ms-(monotonic()-began)*1000)
        p3.need(remaining >= operation_ms, 'insufficient wall budget after E0')
        def backend(op, artifact, lease):
            job = {}
            if registered_config is not None:
                usage = cumulative_usage()
                job.update(operation=op, engine=artifact.digest, reservation_ms=lease,
                           elapsed_ms=round((monotonic()-began)*1000),
                           api_input_before=usage['input'], api_output_before=usage['output_including_reasoning'],
                           node_ms_before=policy.spent_node+cal.node_ms)
            jobs.append(job)
            if registered_config is not None:
                p3.write_once(output/f'development-{len(jobs):02}-start.json', job)
            try:
                result = worker(op, dict(artifact.edits), artifact.digest, base, lease, job)
                if op == 'develop_measure' and result.status == 'observed':
                    performance = p3.exploratory_index(result.data['scenes'], cal.data['scenes'])
                    if registered_config is not None:
                        job['performance_dev'] = performance
                    result = replace(result, data=dict(result.data, performance_dev=performance))
                if truncation_recoveries and op == 'develop_quality':
                    checkpoints.pop(artifact.digest, None)
                    if result.status == 'valid':
                        checkpoints[artifact.digest] = (artifact, result)
                return result
            except (ValueError, RuntimeError, OSError) as exc:
                job['error'] = type(exc).__name__
                raise
            finally:
                if registered_config is not None:
                    job['elapsed_end_ms'] = round((monotonic()-began)*1000)
                p3.write_once(output/f'development-{len(jobs):02}.json', job)

        policy = p3.Session('evolving_train', lineage, narrow, {}, backend, wall_ms=remaining, node_ms=gpu_node_ms-cal.node_ms,
                            dev_calls=development_calls, reservations={'develop_quality':operation_ms, 'develop_measure':operation_ms})
        start_source = base
        start_prompt = prompt
        active_limits = replace(limits, wall_ms=remaining)
        if registered_config is not None:
            policy.deadline = began + limits.wall_ms/1000
        report['stage'] = 'agent'
        for segment in range(1, truncation_recoveries + 2):
            load = snapshot.manifest['load']
            files = dict(snapshot.items)
            fragments = [files[name].decode() for name in load['prompts'] + load['memory']]
            if registered_config is None:
                model = lease_factory(active_limits, transport, counter)
            else:
                model = lease_factory(active_limits, transport, counter, segment=segment,
                                      expected_prompt_fragments=fragments)
                model.deadline = policy.deadline
            models.append(model)
            gateway = p3.Gateway(model, policy, delivery_complete=(lambda: policy.delivered is not None) if stop_on_delivery else None)
            seconds = active_limits.wall_ms//1000
            if registered_config is not None:
                seconds = int(policy.deadline-monotonic())
            p3.need(seconds > 0, 'insufficient wall budget for Pi')
            print(f'Single-function Pi session segment {segment} started', flush=True)
            runner = p3.sandbox(p3.PI, start_source, snapshot, runtime, profile, system, start_prompt, gateway,
                                seconds=seconds, tools=TOOLS, paged_read=True)
            report['runner'] = runner
            clean = (runner['isolated_state_removed'] and runner['socket_removed'] and runner['unchanged_install']
                     and not runner['transport_fault'] and not runner.get('timed_out'))
            truncated = bool(model.rows and model.rows[-1]['result'] == GENERATION_TRUNCATED
                             and model.rows[-1]['usage'] == 'settled')
            context_full = (registered_config is not None and model.rejections
                            and model.rejections[-1]['reason'] == 'context_generation_window'
                            and not model.rejections[-1]['dispatched'])
            resume_cause = GENERATION_TRUNCATED if truncated else 'context_generation_window' if context_full else None
            if registered_config is not None:
                segment_report = dict(segment=segment, start_engine=p3.source_identity(start_source),
                                      start_H=snapshot.identity, limits=asdict(active_limits), runner=runner,
                                      stop_cause=resume_cause,
                                      usage=dict(input=model.spent_input, output_including_reasoning=model.spent_output))
                p3.write_once(output/f'pi-segment-{segment:02}.json', segment_report)
                report['segments'].append(segment_report)
            report['status'] = 'session_completed' if runner['status'] == 'pi_session_complete' and not policy.worker_invalid else 'unsettled'
            if stop_on_delivery and policy.delivered is not None and not policy.worker_invalid and clean:
                report.update(status='agent_delivered', stop_cause='agent_delivery')
                break
            if not resume_cause:
                if registered_config is not None:
                    if model.rejections:
                        report['stop_cause'] = model.rejections[-1]['reason']
                    elif report['status'] == 'session_completed':
                        report['stop_cause'] = 'agent_finished'
                    elif model.rows and model.rows[-1]['result'] != 'complete':
                        report['stop_cause'] = model.rows[-1]['result']
                    elif runner.get('timed_out'):
                        report['stop_cause'] = 'wall_lease'
                break
            report['stop_cause'] = resume_cause
            if segment > truncation_recoveries:
                break
            if not clean or policy.worker_invalid:
                report['recovery_blocked'] = 'session_cleanup_or_worker'
                break
            used = cumulative_usage()
            input_left = limits.input-used['input']
            output_left = limits.output-used['output_including_reasoning']
            requests_left = limits.turns-sum(len(m.rows) for m in models)
            wall_left = int(1000*(policy.deadline-monotonic()))
            if input_left <= 0 or output_left < limits.per_turn_output or requests_left <= 0 or wall_left < 1000:
                report['recovery_blocked'] = 'remaining_budget'
                break
            active_limits = replace(limits, input=input_left, output=output_left, turns=requests_left, wall_ms=wall_left)
            snapshot = lineage.active()
            start_source = base
            recovery = dict(trigger=resume_cause, engine=e0.digest, source_sha=p3.digest(source), harness=snapshot.identity,
                            quality_receipt=None, limits=asdict(active_limits))
            if checkpoints:
                checkpoint, receipt = next(reversed(checkpoints.values()))
                raw = dict(checkpoint.edits)[FILE]
                p3.need(narrow.from_args({'files': {FILE: raw.decode()}}) == checkpoint
                        and policy.qualified.get(checkpoint.digest) == receipt
                        and (receipt.engine, receipt.operation, receipt.status) == (checkpoint.digest, 'develop_quality', 'valid'),
                        'recovery checkpoint does not match its trusted quality receipt')
                with (output/'recovery-gguf.py').open('xb') as fp:
                    fp.write(raw)
                start_source = {FILE: raw}
                recovery.update(engine=checkpoint.digest, source_sha=p3.digest(raw), quality_receipt=receipt.receipt_id)
            p3.write_once(output/'recovery-start.json', recovery)
            report['recovery'] = recovery
            fact = (f'The current file is the last qualified candidate, Engine {recovery["engine"]}, '
                    f'with valid develop_quality receipt {recovery["quality_receipt"]}. '
                    'It can be measured or delivered as-is.' if recovery['quality_receipt'] else
                    'No quality-qualified candidate was available. The current file is the original E0.')
            notice = ('The previous model response reached its generation limit.' if truncated else
                      'The prior conversation could no longer fit a full generation window.')
            start_prompt = (prompt + f'\n\n{notice} This is a fresh Pi process in the same '
                            f'experiment with the remaining original budget and the activated Harness. {fact} '
                            'Continue from the current file. Read only the pages needed for the next action. '
                            'Prior chat and temporary files are unavailable.')
            report.pop('stop_cause', None)
        report['stage'] = 'finished'
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        report['failure'] = type(exc).__name__
        report['failure_detail'] = str(exc)[:220] if type(exc) is ValueError else None
    finally:
        if model is not None:
            report.update(provider=[row for m in models for row in m.rows],
                          provider_rejections=[row for m in models for row in m.rejections],
                          usage=cumulative_usage())
        if policy is not None:
            report.update(rpc=policy.rows, development_calls=development_calls-policy.calls_left,
                          qualified=[r.receipt_id for r in policy.qualified.values() if r.status == 'valid'],
                          node_ms=policy.spent_node+cal.node_ms, final_H=lineage.freeze().identity)
            if policy.delivered is not None:
                raw = dict(policy.delivered.edits)[FILE]
                with (output/'delivered-gguf.py').open('xb') as f:
                    f.write(raw)
                report.update(delivered=policy.delivered.digest, delivered_sha256=p3.digest(raw))
        elif cal is not None:
            report['node_ms'] = cal.node_ms
        elif registered_config is not None and (output/'E0-start.json').exists():
            report['node_usage'] = 'unsettled'
            report['node_ms_reserved'] = operation_ms
        report['wall_ms'] = round((monotonic()-began)*1000)
        p3.write_once(output/'receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    opts = parser.parse_args()
    archive = Path(os.environ['LOCALAPPDATA'])/'EvolveEngine'
    source = (archive/'v3-pinned-sglang-source/gguf.py').read_bytes()
    counter = PublicPromptCounter(archive/'deepseek-flash-1017-08c8657f-offline')
    prereq(source, archive)
    narrow = NarrowContract(source)
    if not opts.execute:
        print(narrow.prompt())
        return
    provider = p3.Provider('https://api.deepseek.com', 'deepseek-flash', 'DEEPSEEK_API_KEY',
                          'DeepSeek-V4.1-Flash', p3.CAP.per_turn_output, fingerprint=FINGERPRINT)
    output = archive/RUN
    p3.need(not output.exists(), 'one-session directory already exists')
    # Read both credentials from a dedicated inherited pipe. Neither is passed to the guest.
    key = sys.stdin.readline(514).rstrip('\r\n')
    password = sys.stdin.readline(514).rstrip('\r\n')
    p3.need(bool(re.fullmatch(r'[A-Za-z0-9_-]{10,512}', key)) and bool(password), 'missing credentials')
    os.environ['DEEPSEEK_API_KEY'] = key
    from unittest.mock import patch
    with patch.object(node_client, 'getpass', lambda _: password), node_client.connection() as ssh:
        response = run(source, output, counter,
                       lambda wire,left: provider.request('/chat/completions',payload=wire,timeout=min(left,180)),
                       make_worker(ssh, archive))
    print(p3.canonical({k:response[k] for k in ('status','stage','qualified','delivered','node_ms','wall_ms','failure')
                        if k in response}).decode(), flush=True)


if __name__ == '__main__':
    main()
