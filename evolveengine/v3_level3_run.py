"""Level 3 exploration driver: trusted worker dispatch and durable Pi handoff."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from math import exp, isfinite, log
import os
from pathlib import Path
from subprocess import SubprocessError
from threading import Event, Lock, Thread
from time import monotonic, sleep

from .journal import Journal
from .v3_level3_artifacts import HarnessAssets
from .v3_harness import Lineage, Snapshot, canonical, digest
from .v3_level3_holdout import register_review
from .v3_level3_lifecycle import Checkpoint, Lifecycle, QualifiedEngine, RECOVERABLE
from .v3_level3_upload import Upload
from .v3_level3_web_research import SCOPE as WEB_SCOPE, WebResearch
from . import v3_level3_scoring as business, v3_level3_weight_audit as weights
from .v3_level3_workload import FIRST_LIVE_PAIR
from .v3_provider import GENERATION_TRUNCATED
from .v3_runner import Gateway, MAX_REPLY, RpcAudit, exception_details, need

EMPTY = dict(prompts=[], skills=[], extensions=[], workflows=[], memory=[])
OPS = frozenset({'develop_build', 'develop_quality', 'develop_measure', 'develop_profile', 'develop_diagnostic'})
CORE = OPS - {'develop_profile', 'develop_diagnostic'}
SCENES = ('W1', 'W2', 'W3', 'W4')
MODE = 'evolving_train'


@dataclass(frozen=True)
class WorkerResult:
    engine: str
    operation: str
    status: str
    node_ms: int
    build: str | None
    evidence: str
    data: dict


def _digest(value):
    return type(value) is str and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def compare_dev(reference, candidate, *, replay_floors=None):
    """Ratio of verified W1–W4 scores. This is not a significance test."""
    need(reference.operation == candidate.operation == 'develop_measure'
         and reference.status == candidate.status == 'observed', 'comparison requires measured evidence')
    pair_keys = {'measurement_inputs_sha256', 'measurement_pair'}
    for key in ('timing_mode','timing_workload_sha256','chat_corpus_sha256','replay_protocol'):
        need(reference.data.get(key) == candidate.data.get(key), 'comparison uses different timing protocols')
    if pair_keys & (reference.data.keys() | candidate.data.keys()):
        need(pair_keys <= reference.data.keys() & candidate.data.keys() and
             _digest(reference.data['measurement_inputs_sha256']) and
             all(reference.data[key] == candidate.data[key] for key in pair_keys),
             'paired timing must use the same independently frozen requests')
    a, b = (r.data.get('scenes') for r in (reference, candidate))
    need(type(a) is dict and type(b) is dict and set(a) == set(b) == set(SCENES),
         'comparison requires the same four workloads')
    ratios = {}
    for scene in SCENES:
        need(type(a[scene]) is dict and type(b[scene]) is dict, 'invalid scene metrics')
        x, y = a[scene].get('dev_score'), b[scene].get('dev_score')
        need(all(type(v) in (int, float) and isfinite(v) and v > 0 for v in (x, y)), 'invalid scene score')
        ratios[scene] = y / x
        need(isfinite(ratios[scene]) and ratios[scene] > 0, 'invalid scene ratio')
    w3 = None
    if reference.data.get('chat_corpus_sha256') and 'w3_requests' in reference.data:
        ratios['W3'],w3 = business.paired_w3_ttft(reference.data['w3_requests'],candidate.data['w3_requests'])
    result = dict(reference=reference.evidence, receipt=candidate.evidence, scene_ratios=ratios,
                  equal_scene_geomean=exp(sum(log(v) for v in ratios.values()) / len(SCENES)),
                  interpretation='descriptive_development_comparison')
    if reference.data.get('timing_mode') == 'natural_eos':
        for arm in (reference, candidate):
            proof = arm.data.get('output_authentication', {})
            from .v3_level3_evidence import BOUND_REPLAY
            method = arm.data.get('replay_protocol', 'independent_same_engine_replay')
            need(method in ('independent_same_engine_replay', BOUND_REPLAY) and proof.get('method') == method and proof.get('status') == 'verified'
                 and _digest(proof.get('replay_receipt')), 'missing independent timing-output replay')
            if method == BOUND_REPLAY:
                detail = proof.get('agreement', {})
                need(type(detail) is dict and set(detail) == set(business.WEIGHTS), 'missing bound replay coverage')
                for scene, d in detail.items():
                    need(all(type(d.get(k)) is int for k in ('requests','exact','same_positions','positions')) and
                         0 <= d['exact'] <= d['requests'] == len(arm.data['timed_request_outputs'][scene]) and
                         0 <= d['same_positions'] <= d['positions'] and d['positions'] >= d['requests'] > 0 and
                         d.get('rate') == d['same_positions']/d['positions'], 'invalid replay agreement accounting')
        if replay_floors is not None:
            rates={s:d['rate'] for s,d in candidate.data['output_authentication']['agreement'].items()}
            result['replay_audit']=dict(floors=replay_floors,rates=rates,below=[s for s in rates if rates[s]<replay_floors[s]],
                                         scoring_effect='none')
        result['output_work'] = business.compare_output_work(reference.data, candidate.data)
    if w3 is not None:
        result['w3_paired_ttft'] = w3
    return result


class Run:
    """Only host-provided, artifact-verifying Worker implementations can issue receipts.

    The worker callable runs one frozen operation and exposes verify(result); the latter
    must revalidate saved artifacts, not just validate the Python receipt's shape.
    """
    def __init__(self, journal, contract, lineage, worker, deadline, *, node_ms, calls, reservations, research=None,
                 require_measurement=False, paired_timing=False, scoring=None, continuous=False, notify=None, weight_audit=None):
        need(CORE <= reservations.keys() <= OPS and all(type(v) is int and v > 0 for v in reservations.values()),
             'worker reservations must cover build, quality and measurement')
        self.reference = None
        self.score_policy = business.resolve_policy(scoring) if scoring else None
        need(self.score_policy is None or paired_timing, 'unsupported scoring or unpaired timing')
        self.scoring = self.score_policy['version'] if self.score_policy else None
        need(weight_audit is None or weight_audit in weights.POLICIES and scoring, 'invalid engineering weight audit')
        self.weight_audit = weight_audit
        self.audit_source_cache = {}
        self.sample_screens = {}
        self.tiered_sampling = bool((self.score_policy or {}).get('replay_audit_floors'))
        self.audited_best_scores = {}
        self.recorded_best = None
        self.continuous = continuous
        self.notify = notify
        self.stage_proposals = []
        self.segment_checkpoint = None
        self.latest_checkpoint = None
        self.historical_facts = None
        self.assets = HarnessAssets(journal.path.parent / 'harness-assets') if scoring else None
        self.require_measurement = require_measurement
        self.paired_timing, self.next_timing_pair, self.e0 = paired_timing, FIRST_LIVE_PAIR, None
        self.measurements = {}
        self.journal, self.contract, self.lineage, self.worker = journal, contract, lineage, worker
        self.research = research
        self.deadline, self.left, self.calls, self.reservations = deadline, node_ms, calls, reservations
        self.engines, self.qualified, self.parents = {}, {}, {}
        self.qualified_builds = {}
        self.receipts, self.builds, self.facts, self.drafts = {}, {}, {}, {}
        self.jobs, self.events = [], []
        self.worker_invalid, self.pending = False, None
        self.rpc_unsettled = False
        self.lifecycle = Lifecycle(journal, verify_artifact=self._artifact, verify_quality=self._quality,
                                   verify_measure=self._measure, verify_wip=self._wip, max_recoveries=None if deadline is None else 1)

    def _event(self, op, **data):
        row = dict(operation=op, **data)
        self.events.append(row)
        self.journal.event(f'dispatch/{len(self.events):06}', row)
        if self.continuous and op in {'segment_started','segment_settled','worker_reserved','worker_settled',
                                     'checkpoint_saved','business_best_updated','stage_proposal','exploration_paused',
                                     'infrastructure_backoff','stagnation_notice','operator_interrupted'}:
            report = self.journal.path.parent / 'status.json'
            temporary = report.with_suffix('.tmp')
            temporary.write_bytes(canonical(dict(event=row, last_checkpoint=self.latest_checkpoint,
                                                  phase_proposals=len(self.stage_proposals),
                                                  latest_proposal=self.stage_proposals[-1] if self.stage_proposals else None,
                                                  closed_jobs=sum(j.get('node_ms') is not None for j in self.jobs)))+b'\n')
            temporary.replace(report)

    def _artifact(self, kind, identity):
        if kind == 'harness':
            return self.lineage.get(identity).identity == identity
        if kind == 'engine':
            candidate = self.engines.get(identity)
            return candidate is not None and self.contract.from_args(self.contract.to_args(candidate)).digest == identity
        if kind in {'build', 'quality'}:
            result = self.receipts.get(identity)
            operation = 'develop_build' if kind == 'build' else 'develop_quality'
            return result is not None and result.operation == operation and self.worker.verify(result) is True
        if kind == 'trusted_facts':
            return identity in self.facts and digest(canonical(self.facts[identity])) == identity
        if kind == 'harness_memory':
            return self.journal.get('level3_harness_memory', identity) is not None
        if kind == 'research' and isinstance(self.research, WebResearch):
            return self.research.read(identity).get('identity') == identity
        if kind == 'wip':
            return identity in self.drafts and self._artifact('engine', self.drafts[identity][1])
        return False

    def _quality(self, artifact):
        r = self.receipts.get(artifact.quality)
        return (r is not None and (r.engine, r.build, r.operation, r.status) ==
                (artifact.engine, artifact.build, 'develop_quality', 'valid'))

    def _measure(self, artifact, identity):
        r = self.receipts.get(identity)
        return (r is not None and (r.engine, r.build, r.operation, r.status) ==
                (artifact.engine, artifact.build, 'develop_measure', 'observed') and self.worker.verify(r) is True)

    def _wip(self, reference, parent):
        return reference in self.drafts and self.drafts[reference][0] == parent

    def add_engine(self, engine, parent=None):
        self.engines[engine.digest] = engine
        self.parents.setdefault(engine.digest, set()).add(parent)
        source = self.journal.put('level3_engine', self.contract.to_args(engine))
        self._event('engine_registered', engine=engine.digest, parent=parent, source_object=source)
        return engine.digest

    def _can_dispatch(self, cap, count=1):
        return ((self.calls is None or self.calls >= count) and
                (self.left is None or self.left >= cap * count) and
                (self.deadline is None or (self.deadline-monotonic())*1000 >= cap * count))

    def develop(self, op, engine, *, diagnostic_source=None, timing_pair=None, reference=None, independent=False,
                profile_workload=None, stress=False, sample=None, sample_build=None):
        need(op in self.reservations, 'operation not available in this run')
        need((op == 'develop_diagnostic') == (type(diagnostic_source) is str),
             'diagnostic source is required only for a diagnostic run')
        need((op == 'develop_profile') == (type(profile_workload) is dict), 'profiling requires a workload')
        need((timing_pair is not None) == (self.paired_timing and op == 'develop_measure'), 'invalid timing pair request')
        cap = self.reservations[op]
        need(not self.worker_invalid and self._can_dispatch(cap), 'worker lease unavailable')
        build = self.builds.get(engine.digest)
        if sample is not None:
            need(op=='develop_quality' and self.tiered_sampling and sample_build is not None,'sampling is a trusted milestone operation')
            build=self.receipts[sample_build]
            need(build.engine==engine.digest and build.operation=='develop_build' and build.status=='built','unbound milestone build')
        if independent:
            need(op == 'develop_build' and self.scoring, 'independent verification requires interactive exploration')
            build = None
        elif op == 'develop_build' and build is None and self.scoring:
            ancestors, visited = self.parents.get(engine.digest, set()) - {None}, {engine.digest}
            while ancestors - visited:
                level = sorted(ancestors - visited)
                build = next((self.builds[item] for item in level if item in self.builds), None)
                if build is not None:
                    break
                visited.update(level)
                ancestors = set().union(*(self.parents.get(item, set()) for item in level)) - {None}
        need(op == 'develop_build' or build is not None, 'candidate must be built before this operation')
        qualified = self.qualified.get(engine.digest)
        need(op != 'develop_measure' or qualified is not None and qualified.build == build.evidence,
             'scoring requires quality acceptance for the loaded build')
        if self.left is not None:
            self.left -= cap
            self.calls -= 1
        row = dict(operation=op, engine=engine.digest, reserved_ms=cap, status='unsettled')
        if timing_pair is not None:
            row['measurement_pair'] = timing_pair
        if sample is not None: row['sampling_challenge']=sample
        if stress:
            row['fixed_output_stress'] = True
        self.jobs.append(row)
        self._event('worker_reserved', **{k:v for k,v in row.items() if k != 'operation'}, worker_operation=op)
        try:
            if sample is not None:
                result=self.worker.sampling(engine,build,cap,sample)
                need(result.data.get('challenge')==sample and result.status in ('sampled','insufficient_evidence'),'unbound milestone draw')
            elif op == 'develop_diagnostic':
                diagnose = getattr(self.worker, 'diagnose', None)
                need(callable(diagnose), 'Worker has no registered non-scoring diagnostic lane')
                result = diagnose(engine, build, diagnostic_source, cap)
            elif op == 'develop_profile':
                result = self.worker.profile(engine, build, cap, profile_workload)
            elif stress:
                result = self.worker.stress(engine, build, cap, timing_pair)
            elif timing_pair is not None:
                result = self.worker.measure(engine, build, cap, timing_pair)
            elif independent:
                result = self.worker.independent_build(engine, cap)
            else:
                result = self.worker(op, engine, build, cap)
            need(type(result) is WorkerResult and (result.engine, result.operation) == (engine.digest, op)
                 and type(result.node_ms) is int and 0 <= result.node_ms <= cap and _digest(result.evidence)
                 and (op == 'develop_build' and result.build is None or op != 'develop_build' and result.build == build.evidence)
                 and self.worker.verify(result) is True, 'invalid or unverified worker result')
            if independent and result.status == 'built':
                need(result.data.get('build_mode') == 'independent_cold', 'independent verification reused a build')
            if timing_pair is not None and result.status == 'observed':
                need(result.data.get('measurement_pair') == timing_pair and
                     _digest(result.data.get('measurement_inputs_sha256')) and
                     _digest(result.data.get('timed_outputs_sha256')), 'missing frozen timing inputs or outputs')
                if self.scoring in business.NATURAL_VERSIONS and not stress:
                    need(result.data.get('timing_mode') == 'natural_eos' and
                         result.data.get('timing_workload_sha256') == self.score_policy['timing_workload_sha256'],
                         'timing does not use the calibrated natural-EOS workload')
        except Exception as e:
            row['error'] = type(e).__name__
            self.worker_invalid = True
            self._event('worker_uncertain', engine=engine.digest, worker_operation=op, error=type(e).__name__)
            raise RuntimeError('worker evidence unavailable; reserved charge retained') from e
        prior = self.receipts.get(result.evidence)
        need(prior is None or prior == result, 'worker reused a receipt identity with different contents')
        if self.left is not None:
            self.left += cap - result.node_ms
        self.receipts[result.evidence] = result
        row.update(asdict(result))
        saved = self.journal.put('level3_worker_result', asdict(result))
        self._event('worker_settled', evidence=result.evidence, result_object=saved)
        if op == 'develop_build' and result.status == 'built':
            self.builds[engine.digest] = result
        if op == 'develop_quality' and result.status == 'valid':
            if self.weight_audit:
                need(bool(result.data.get('weight_observation')), 'Worker did not collect registered weight telemetry')
            artifact = QualifiedEngine(engine.digest, result.build, result.evidence)
            self.lifecycle.record_quality_pass(artifact)
            self.qualified[engine.digest] = artifact
            self.qualified_builds[engine.digest, result.build] = artifact
        if op == 'develop_measure' and result.status == 'observed' and not stress:
            comparison_base = reference if self.paired_timing else self.reference
            if reference is not None and result.data['measurement_inputs_sha256'] != reference.data['measurement_inputs_sha256']:
                self.worker_invalid = True
                self._event('development_timing_input_mismatch', engine=engine.digest,
                            reference=reference.evidence, candidate=result.evidence)
                raise RuntimeError('Worker compared different timing requests')
            if (reference is not None and self.scoring not in business.NATURAL_VERSIONS and
                    result.data['timed_outputs_sha256'] != reference.data['timed_outputs_sha256']):
                row['scoring_status'] = 'timed_output_mismatch'
                self._event('development_timed_output_mismatch', engine=engine.digest,
                            reference=reference.evidence, candidate=result.evidence)
                self.measurements.pop((engine.digest, result.build), None)
                return dict(status='timed_output_mismatch', engine=engine.digest,
                            evidence=result.evidence, reference=reference.evidence)
            comparison = compare_dev(comparison_base, result,replay_floors=(self.score_policy or {}).get('replay_audit_floors')) if comparison_base is not None else None
            self.lifecycle.record_measure(qualified, result.evidence)
            self.measurements[engine.digest, result.build] = result.evidence
            if comparison is not None:
                row['performance'] = comparison
                self._event('development_comparison', engine=engine.digest, **row['performance'])
        return asdict(result) | ({'performance': row['performance']} if 'performance' in row else {})

    def measure_pair(self, engine):
        op = 'develop_measure'
        cap = self.reservations[op]
        build, qualified = self.builds.get(engine.digest), self.qualified.get(engine.digest)
        need(build is not None and qualified is not None and qualified.build == build.evidence,
             'scoring requires quality acceptance for the loaded build')
        need(self._can_dispatch(cap, count=2),
             'node, wall or call lease cannot cover both runs in the comparison')
        pair = self.next_timing_pair
        self.next_timing_pair += 1
        need(self.e0.digest in self.qualified, 'paired reference has no verified quality')
        if self.scoring:
            summary = self.business_summary(engine.digest)
            candidate_first = summary['candidate_first'] * 2 < summary['pairs']
            arms = (engine, self.e0) if candidate_first else (self.e0, engine)
            rows = []
            for arm in arms:
                observed = self.develop(op, arm, timing_pair=pair)
                if observed['status'] != 'observed':
                    if observed['status'] == 'output_replay_mismatch':
                        self.jobs[-1]['scoring_status'] = 'output_replay_mismatch'
                        self._event('development_output_replay_mismatch', engine=arm.digest, candidate=observed['evidence'])
                        return dict(status='output_replay_mismatch', evidence=observed['evidence'])
                    return dict(status='insufficient_evidence', evidence=observed['evidence'])
                rows.append(self.jobs[-1])
            baseline, candidate = reversed(rows) if candidate_first else rows
            base, trial = (self.receipts[r['evidence']] for r in (baseline, candidate))
            if base.data['measurement_inputs_sha256'] != trial.data['measurement_inputs_sha256']:
                self.worker_invalid = True
                self._event('development_timing_input_mismatch', engine=engine.digest,
                            reference=base.evidence, candidate=trial.evidence)
                raise RuntimeError('Worker compared different timing requests')
            comparison = compare_dev(base, trial,replay_floors=(self.score_policy or {}).get('replay_audit_floors'))
            if self.scoring in business.NATURAL_VERSIONS:
                if not all(scene['comparable'] for scene in comparison['output_work'].values()):
                    candidate['scoring_status'] = 'workload_drift'
                    self._event('development_workload_drift', engine=engine.digest, candidate=trial.evidence,
                                reference=base.evidence, output_work=comparison['output_work'])
                    return dict(status='workload_drift', evidence=trial.evidence, output_work=comparison['output_work'])
            elif base.data['timed_outputs_sha256'] != trial.data['timed_outputs_sha256']:
                candidate['scoring_status'] = 'timed_output_mismatch'
                self.measurements.pop((engine.digest, trial.build), None)
                self._event('development_timed_output_mismatch', engine=engine.digest,
                            reference=base.evidence, candidate=trial.evidence)
                return dict(status='timed_output_mismatch', evidence=trial.evidence, reference=base.evidence)
            comparison.update(
                scoring_version=self.scoring,
                weighted_score=business.weighted_score(comparison['scene_ratios'], self.score_policy['weight_percent']),
            )
            if self.scoring == business.WORKBUDDY_VERSION:
                sla = business.workbuddy_sla_summary(trial.data.get('scenes') or {})
                if sla:
                    comparison['workbuddy_sla'] = sla
            if self.weight_audit:
                comparison['gain_status'] = weights.PROVISIONAL
            candidate['performance'] = comparison
            self.measurements[engine.digest, trial.build] = trial.evidence
            self._event('development_comparison', engine=engine.digest, **comparison)
            if self.weight_audit:
                record = self.historical_best()
                if record and record['weighted_score'] > self.audited_best_scores.get((record['engine'], record['build']), float('-inf')):
                    self.audit_candidate('historical_best', self.qualified_builds[record['engine'], record['build']],
                                         score=record['weighted_score'])
                    self.audited_best_scores[record['engine'],record['build']] = record['weighted_score']
            best = self.business_best()
            if best != self.recorded_best:
                self._event('business_best_updated', best=best)
                self.recorded_best = best
            return dict(**asdict(trial), performance=comparison, qualification=self.business_summary(engine.digest))
        baseline = self.develop(op, self.e0, timing_pair=pair)
        if baseline['status'] != 'observed':
            return dict(status='insufficient_evidence', reason='paired_reference_not_observed', reference=baseline['evidence'])
        return self.develop(op, engine, timing_pair=pair, reference=self.receipts[baseline['evidence']])

    def confirm_performance(self, engine, cancelled=None):
        summary = self.business_summary(engine.digest)
        if summary['status'] in {'eligible', 'regression', 'timed_output_mismatch', 'workload_drift','weight_audit_rejected'}:
            return summary
        count = max(0, business.MIN_PAIRS - summary['pairs'])
        if count == 0:
            count = 1 if summary['pairs'] % 2 else 2
        for _ in range(count):
            if cancelled is not None and cancelled.is_set():
                break
            response = self.measure_pair(engine)
            if response['status'] != 'observed':
                break
        return self.business_summary(engine.digest)

    def business_summary(self, engine):
        if not self.scoring:
            return None
        build = self.builds.get(engine)
        summary=business.summarize_build(self.jobs, engine, build.evidence if build else '',
                                         self.score_policy if self.scoring in business.NATURAL_VERSIONS else self.scoring)
        if self.weight_audit:
            summary['gain_status']=weights.PROVISIONAL
            if build and (screen:=self.sample_screens.get((engine,build.evidence,'historical_best'))):
                summary['historical_sampling']=screen
            if self.sampling_rejected(engine): summary.update(status='sampling_different',eligible_for_delivery=False)
            if build and self.weight_rejected(engine,build.evidence):
                summary.update(status='weight_audit_rejected',eligible_for_delivery=False,gain_status='rejected')
        return summary

    def sampling_rejected(self, engine):
        # A fresh binary is not a fresh hypothesis about the same submitted Engine.
        return any(k[0] == engine and r['decision'] == 'different' for k,r in self.sample_screens.items())

    def weight_rejected(self, engine, build):
        return self.weight_audit and (self.journal.path.parent/'weight-audit/rejections'/engine/build).is_file()

    def business_best(self):
        eligible = []
        for engine, quality in self.qualified.items():
            if self.builds[engine].evidence != quality.build:
                continue
            summary = self.business_summary(engine)
            if summary and summary['eligible_for_delivery']:
                eligible.append((summary['weighted_score'], engine))
        if eligible:
            score, engine = max(eligible)
            return dict(**({'gain_status': weights.PROVISIONAL} if self.weight_audit else {}),
                        engine=engine, build=self.builds[engine].evidence,
                        quality=self.qualified[engine].quality, weighted_score=score)
        return None

    def milestone_screen(self, chosen, mark):
        """No free redraws after failure or across segments. Final gets fresh draws, even on a sampled best."""
        from .v3_level3_sampling import VERSION
        if not self.tiered_sampling: return None
        key=(chosen.engine,chosen.build,mark)
        if key in self.sample_screens: return self.sample_screens[key]
        # A rebuilt byte-identical source cannot fish for a pass by repeating the same milestone.
        if any(k[0]==chosen.engine and k[2]==mark for k in self.sample_screens):
            return dict(protocol=VERSION,decision='already_attempted_other_build',receipts=[])
        report=dict(protocol=VERSION,decision='incomplete',receipts=[])
        self.sample_screens[key]=report
        for stage in (1,2):
            reference=None
            for role in ('reference','trial'):
                challenge=dict(milestone=mark,stage=stage,role=role)
                obs=self.develop('develop_quality',self.engines[chosen.engine],sample=challenge,sample_build=chosen.build)
                report['receipts'].append(obs['evidence']);data=obs['data']
                if obs['status']!='sampled': return report
                if role=='reference': reference=obs['evidence']
                else:
                    need(data['reference']==reference and data['decision'] in ('pass','inconclusive','different'),
                         'milestone trial not bound to independent reference')
                    report['decision']=data['decision']
            if report['decision']!='inconclusive': break
            if stage==1: report['decision']='incomplete'
        self._event('milestone_sampling',engine=chosen.engine,build=chosen.build,milestone=mark,**report)
        return report

    def audit_candidate(self, trigger, chosen, **context):
        if not self.weight_audit:
            return None
        if self.tiered_sampling and trigger in ('historical_best','final_nomination'):
            context['sampling']=self.milestone_screen(chosen,trigger)
            if trigger=='final_nomination': need(context['sampling']['decision']=='pass','independent final sampling not passed')
        base = self.qualified[self.e0.digest]
        jobs = [row for row in self.jobs if row.get('build') == chosen.build and row['engine'] == chosen.engine and row.get('performance')]
        key = weights.packet(self.journal.path.parent/'weight-audit', self.contract, self.engines[chosen.engine],
                             asdict(chosen), asdict(self.receipts[chosen.build]), asdict(self.receipts[chosen.quality]),
                             asdict(self.receipts[base.quality]), jobs, trigger=trigger, context=context,
                             source_cache=self.audit_source_cache, policy=self.weight_audit)
        self._event('weight_audit_requested', trigger=trigger, packet=key, chosen=asdict(chosen), **context)
        return key

    def historical_best(self):
        if not self.scoring:
            return None
        eligible = []
        for (engine, build), quality in self.qualified_builds.items():
            if self.weight_rejected(engine,build) or self.sampling_rejected(engine): continue
            summary = business.summarize_build(self.jobs, engine, build, self.score_policy
                                               if self.scoring in business.NATURAL_VERSIONS else self.scoring)
            if summary['eligible_for_delivery']:
                eligible.append((summary['weighted_score'], engine, build, quality.quality))
        if eligible:
            score, engine, build, quality = max(eligible)
            return dict(engine=engine, build=build, quality=quality, weighted_score=score,
                        **({'gain_status': weights.PROVISIONAL} if self.weight_audit else {}))
        return None

    @staticmethod
    def _public_job(row):
        return {key: row[key] for key in ('operation', 'engine', 'status', 'evidence', 'performance') if key in row}

    def checkpoint(self, engine, wip=None):
        need(engine in self.qualified, 'checkpoint requires quality-approved Engine')
        if wip is not None:
            need(wip in self.engines and engine in self.parents[wip], 'draft must be submitted against the checkpoint Engine')
            draft = dict(base=engine, candidate=wip)
            ref = self.journal.put('level3_wip', draft)
            self.drafts[ref] = engine, wip
        else:
            ref = None
        rows = [self._public_job(row) for row in self.jobs]
        if self.assets and self.assets.assets.files:
            rows.append(dict(operation='harness_asset_catalogue', identities=sorted(self.assets.assets.files)))
        address = digest(canonical(rows))
        self.facts[address] = rows
        self.journal.put('level3_facts', rows)
        harness = self.lineage.active()
        memory_files = dict(harness.items)
        memory = tuple(self.journal.put('level3_harness_memory',
                                      dict(path=name,value=memory_files[name].decode('utf-8')))
                       for name in harness.manifest['load']['memory']) if self.continuous else ()
        research = self.research.page_refs() if self.continuous and isinstance(self.research, WebResearch) else ()
        state = Checkpoint(self.qualified[engine], harness.identity, address, harness_memory=memory, research=research, wip=ref)
        cp = self.lifecycle.save(state)
        if self.scoring:
            self._export_checkpoint(cp, state, rows)
        self.audit_candidate('checkpoint', state.engine, checkpoint=cp, wip=wip)
        self.latest_checkpoint = cp
        self._event('checkpoint_saved', checkpoint=cp, engine=engine, wip=wip)
        return cp

    def continuation_checkpoint(self, fallback):
        state = self.lifecycle.load(self.segment_checkpoint or fallback)
        wip = self.drafts[state.wip][1] if state.wip else None
        return self.checkpoint(state.engine.engine, wip)

    def _export_checkpoint(self, identity, state, facts):
        """Portable resume input. The next Worker independently qualifies these source bytes."""
        folder = self.journal.path.parent / 'checkpoint-exports'
        folder.mkdir(exist_ok=True)
        path = folder / (identity + '.json')
        harness = self.lineage.get(state.harness)
        payload = dict(schema='level3-resume-checkpoint-v1', checkpoint=identity,
                       engine=self.contract.to_args(self.engines[state.engine.engine]),
                       engine_id=state.engine.engine,
                       harness=dict(files={k: v.decode('utf-8') for k, v in harness.items},
                                    load=harness.manifest['load']),
                       wip=self.contract.to_args(self.engines[self.drafts[state.wip][1]]) if state.wip else None,
                       asset_identities=sorted(self.assets.assets.files), research_identities=list(state.research),
                       recent_facts=facts[-16:])
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as out:
                out.write(canonical(payload) + b'\n')
                out.flush()
                os.fsync(out.fileno())
            temporary.replace(path)
        pointer = folder / 'latest.tmp'
        pointer.write_text(path.name + '\n', encoding='ascii')
        pointer.replace(folder/'latest')

    @contextmanager
    def handler(self, segment, state, *, audit=None):
        parent = self.drafts[state.wip][1] if state.wip else state.engine.engine
        jobs = {}
        active = None
        background = None
        synchronous_worker = Event()
        worker_routes = OPS | {'develop_stress','confirm_performance','confirm_cold_build','diagnostic_submit'}

        def process(op, args):
            nonlocal active, background
            if op in {'job_status','job_cancel'} and self.scoring:
                need(set(args) == {'job'} and args['job'] in jobs, 'unknown segment job')
                job = jobs[args['job']]
                with job['lock']:
                    if op == 'job_cancel' and job['status'] == 'running':
                        if job['operation'] != 'confirm_performance':
                            return 409, dict(status='single_worker_operation_in_flight')
                        job['cancel'].set()
                    return 200, {k: v for k, v in job.items() if k not in {'lock','cancel'}}
            need(self.pending is None and not self.worker_invalid and (self.deadline is None or monotonic() < self.deadline), 'segment ended or unavailable')
            busy = background is not None and background.is_alive()
            if busy:
                return 409, dict(status='worker_busy', job=active)
            if op == 'submit_job' and self.scoring:
                need(set(args) == {'operation','arguments'} and args['operation'] in
                     (worker_routes - {'develop_diagnostic'})
                     and type(args['arguments']) is dict, 'unsupported background job')
                name, values = args['operation'], args['arguments']
                active = f'{segment}-{len(jobs)+1}'
                job = dict(job=active, operation=name, status='running', lock=Lock(), cancel=Event())
                jobs[active] = job

                def execute_job():
                    start = monotonic()
                    status, result = 500, dict(error='BackgroundJobAborted')
                    detail = {}
                    try:
                        if audit is not None:
                            audit(event='background_started',tool=name,job=job['job'])
                        code, reply = self._rpc(name, values, segment, parent, cancelled=job['cancel'])
                        if type(code) is not int or not 100 <= code <= 599 or len(canonical(reply)) > MAX_REPLY-1024:
                            raise RuntimeError('invalid background job response envelope')
                        status, result = code, reply
                    except (ValueError, KeyError, TypeError, PermissionError) as exc:
                        status, result = 400, dict(error=type(exc).__name__)
                    except Exception as exc:
                        status, result = 500, dict(error=type(exc).__name__)
                        detail = exception_details(exc)
                    finally:
                        with job['lock']:
                            job.update(status='settled' if 200 <= status < 300 else 'failed', http_status=status,
                                       result=result, stop_requested=job['cancel'].is_set())
                        if audit is not None:
                            audit(event='background_finished',tool=name,job=job['job'],status=status,
                                  duration_ms=round((monotonic()-start)*1000),**detail)

                background = Thread(target=execute_job, name=f'level3-worker-{active}')
                try:
                    background.start()
                except Exception as exc:
                    job.update(status='failed', http_status=500, result=dict(error=type(exc).__name__))
                    background = None
                    raise
                return 202, dict(job=active, operation=name, status='running')
            if op in worker_routes:
                synchronous_worker.set()
            try:
                return self._rpc(op, args, segment, parent)
            except (ValueError, KeyError, TypeError, PermissionError) as e:
                self._event('tool_refused', tool=op, error=type(e).__name__)
                return 400, dict(status='denied', error=str(e)[:240] if type(e) is ValueError else type(e).__name__)
            finally:
                synchronous_worker.clear()
        upload = Upload(process, objective=bool(self.scoring))
        lock = Lock()
        closed = Event()

        def dispatch(op, args):
            with lock:
                need(not closed.is_set() and self.pending is None and (not self.worker_invalid or op == 'job_status')
                     and (self.deadline is None or monotonic() < self.deadline), 'segment ended or unavailable')
                return upload(op, args)

        def worker_active():
            job_thread = background
            return synchronous_worker.is_set() or (job_thread is not None and job_thread.is_alive())

        dispatch.worker_active = worker_active
        try:
            yield dispatch
        finally:
            # Revoke new calls immediately. An arbitrary stuck host callback
            # cannot be killed in-process; report uncertainty instead of hanging
            # here or handing off while it can still mutate the old segment.
            closed.set()
            if lock.acquire(timeout=1):
                try:
                    upload.close()
                finally:
                    lock.release()
            else:
                self.rpc_unsettled = True
            if background is not None:
                background.join()

    def _rpc(self, op, args, segment, parent, *, cancelled=None):
        if op == 'dependency_fetch':
            need(self.continuous and set(args) == {'name','version'}, 'pinned npm dependencies require objective mode')
            from .v3_level3_dependencies import fetch
            receipt = fetch(self.assets,**args)
            self._event('dependency_fetch', **receipt)
            return 200, receipt
        if op in {'harness_asset_put','harness_asset_read'}:
            need(self.assets is not None, 'assets require interactive exploration')
            if op == 'harness_asset_read':
                return 200, self.assets.read(**args)
            result = self.assets.put(**args)
            if result['committed']:
                self._event('harness_asset_saved', identity=result['identity'], size=result['size'])
            return 200, result
        if op == 'confirm_cold_build':
            need(self.scoring and set(args) == {'engine'}, 'independent verification requires an interactive candidate digest')
            engine = self.engines[args['engine']]
            build = self.builds[engine.digest]
            if build.data.get('build_mode') == 'independent_cold':
                return 200, asdict(build)
            summary = self.business_summary(engine.digest)
            if not summary['eligible_for_delivery']:
                return 409, dict(status='business_qualification_required', qualification=summary)
            return 200, self.develop('develop_build', engine, independent=True)
        if op == 'confirm_performance':
            need(self.scoring and set(args) == {'engine'}, 'confirmation requires an interactive candidate digest')
            return 200, self.confirm_performance(self.engines[args['engine']], cancelled)
        if op == 'artifact_query':
            need('view' in args and set(args) <= {'view','identity','offset','limit'}, 'invalid artifact query')
            response = self.worker.artifact_query(**args)
            self._event(op, view=args['view'], identity=args.get('identity'), response_sha256=digest(canonical(response)))
            return 200, response
        if op == 'diagnostic_submit':
            diagnostic_max = 2 << 20 if self.scoring else 24 << 10
            need(set(args) == {'engine', 'source'} and type(args['engine']) is str
                 and type(args['source']) is str and 0 < len(args['source'].encode('utf-8')) <= diagnostic_max
                 and '\0' not in args['source'], 'diagnostic requires one built Engine and one bounded script')
            return 200, self.develop('develop_diagnostic', self.engines[args['engine']],
                                     diagnostic_source=args['source'])
        if op == 'develop_profile':
            need({'engine'} <= args.keys() <= {'engine','scenes','windows','inputs'}, 'invalid profile request')
            return 200, self.develop(op, self.engines[args['engine']],
                                     profile_workload={k:v for k,v in args.items() if k != 'engine'})
        if op == 'develop_stress':
            need(self.scoring in business.NATURAL_VERSIONS and set(args) == {'engine'}, 'invalid stress request')
            index = self.next_timing_pair
            self.next_timing_pair += 1
            return 200, dict(**self.develop('develop_measure', self.engines[args['engine']],
                                            timing_pair=index, stress=True), scoring=False)
        if op in OPS:
            need(set(args) == {'engine'}, 'development requires an engine digest')
            candidate = self.engines[args['engine']]
            if self.paired_timing and candidate.digest == self.e0.digest and op == 'develop_build' and candidate.digest in self.builds:
                return 200, asdict(self.builds[candidate.digest])
            if op == 'develop_measure' and self.paired_timing:
                return 200, self.measure_pair(candidate)
            return 200, self.develop(op, candidate)
        if op == 'candidate_submit':
            base = args.get('parent', parent)
            changes = {k:v for k,v in args.items() if k != 'parent'}
            candidate = self.contract.from_parent(self.engines[base], changes)
            return 200, dict(engine=self.add_engine(candidate, base), parent=base)
        if op == 'harness_submit':
            need(set(args) == {'files', 'load'} and type(args['files']) is dict
                 and all(type(v) is str for v in args['files'].values()), 'invalid Harness upload')
            version = self.lineage.submit(MODE, {k:v.encode() for k,v in args['files'].items()}, args['load'])
            self._event(op, version=version)
            return 200, dict(version=version)
        if op == 'harness_activate':
            need(set(args) == {'version', 'expected'}, 'activation requires an expected version')
            row = self.lineage.activate(MODE, args['version'], args['expected'], actor='agent_rpc',
                                        request_id=f'level3-{segment}-{len(self.events)}')
            self._event(op, **row)
            return 200, row
        if op == 'harness_status':
            need(not args, 'harness_status takes no arguments')
            return 200, dict(active=self.lineage.active().identity, versions=self.lineage.catalogue())
        if op == 'facts':
            need(not args, 'facts takes no arguments')
            if isinstance(self.research, WebResearch):
                research_guide = dict(
                    scope=WEB_SCOPE, live_web=True, cached_available=self.research.cache is not None,
                    operations={
                        'research_search': '{query, count? (1..50), cached?}',
                        'research_fetch': '{url} (fetches public HTTP/HTTPS URL and returns page 1 directly)',
                        'research_read': '{identity, offset} (only call when next_offset is non-null)',
                        'research_reuse': '{kind: original|technology_transfer|upstream_patch, paths, url?, revision?, license?, identity?}',
                    },
                )
            elif self.research is not None:
                research_guide = dict(
                    scope='reviewed_public_cache', live_web=False, cached_available=True,
                    operations={
                        'research_search': '{query, limit?, offset?}',
                        'research_read': '{identity, offset?}',
                    },
                )
            else:
                research_guide = None
            return 200, dict(**(dict(business_policy=self.score_policy, best_eligible=self.business_best(),
                                    historical_best=self.historical_best(), latest_checkpoint=self.latest_checkpoint,
                                    latest_proposal=self.stage_proposals[-1] if self.stage_proposals else None,
                                    prior_run_facts=self.historical_facts) if self.scoring else {}),
                             workspace_base=parent, active_harness=self.lineage.active().identity,
                             qualified=list(self.qualified), recent_jobs=[self._public_job(job) for job in self.jobs[-8:]],
                             remaining_node_ms=self.left, available_worker_operations=sorted(self.reservations),
                             development_reference=self.reference.evidence if self.reference else None,
                             research_scope=WEB_SCOPE if isinstance(self.research, WebResearch) else
                             'reviewed_public_cache' if self.research else None,
                             research_guide=research_guide)
        if op in {'research_search','research_fetch','research_read','research_reuse'} and isinstance(self.research, WebResearch):
            call = {'research_search': self.research.search, 'research_fetch': self.research.fetch,
                    'research_read': self.research.read, 'research_reuse': self.research.record_reuse}[op]
            result = call(**args)
            self._event(op, request_sha256=digest(canonical(args)), response_sha256=digest(canonical(result)),
                        live_call=result.get('call'), receipt=result.get('receipt'))
            return 200, result
        if op in {'research_search', 'research_read'}:
            need(self.research is not None, 'no registered public reference corpus')
            if op == 'research_search':
                need({'query'} <= args.keys() <= {'query', 'limit', 'offset'}, 'search requires a query')
                data = self.research.search(**args)
                refs = [item['sha256'] for item in data['results']]
            else:
                need({'identity'} <= args.keys() <= {'identity', 'offset'}, 'read requires an admitted content digest')
                data = self.research.read(**args)
                refs = [data['source']['sha256']]
            self._event(op, corpus=self.research.identity, request_sha256=digest(canonical(args)), results=refs,
                        response_bytes=len(canonical(data)), retrieval='local_cache', charge_rmb='0')
            return 200, data
        if op == 'save_checkpoint':
            need({'engine'} <= args.keys() <= {'engine', 'wip'}, 'invalid checkpoint request')
            checkpoint = self.checkpoint(args['engine'], args.get('wip'))
            self.segment_checkpoint = checkpoint
            return 200, dict(checkpoint=checkpoint)
        if op == 'continue_session':
            need(set(args) == {'checkpoint'}, 'continue requires a saved checkpoint')
            cp = self.lifecycle.load(args['checkpoint'])
            need(cp.harness == self.lineage.active().identity, 'checkpoint Harness is not active')
            self.pending = 'continue', args['checkpoint']
            return 200, dict(status='continue_after_cleanup')
        if op == 'finish_exploration':
            continuous = getattr(self, 'continuous', False)
            need({'engine'} <= args.keys() <= ({'engine','checkpoint'} if continuous else {'engine'}) and
                 args['engine'] in self.qualified, 'finish requires an explicitly chosen legal Engine')
            chosen = self.qualified[args['engine']]
            if self.scoring:
                need(self.builds[chosen.engine].evidence == chosen.build, 'latest build needs quality acceptance')
                summary = self.business_summary(chosen.engine)
                if not summary['eligible_for_delivery']:
                    return 409, dict(status='business_qualification_required', qualification=summary)
                if not continuous and self.builds[chosen.engine].data.get('build_mode') != 'independent_cold':
                    return 409, dict(status='independent_cold_build_required')
            need(not self.require_measurement or (chosen.engine, chosen.build) in self.measurements,
                 'delivery requires verified development timing of the selected build')
            if self.tiered_sampling:
                if self.builds[chosen.engine].data.get('build_mode')!='independent_cold':
                    return 409,dict(status='independent_cold_build_required')
                final_sampling=self.milestone_screen(chosen,'final_nomination')
                if final_sampling['decision']!='pass': return 409,dict(status='final_sampling_not_passed',sampling=final_sampling)
            if continuous:
                if 'checkpoint' in args:
                    self.segment_checkpoint = args['checkpoint']
                prior = self.journal.get_event(f'level3/segment/{segment}/start')['checkpoint']
                checkpoint = self.continuation_checkpoint(prior)
                self.pending = 'nominate', (chosen, checkpoint, summary)
                return 200, dict(status='proposal_then_continue', checkpoint=checkpoint)
            self.pending = 'deliver', chosen
            return 200, dict(status='seal_after_cleanup')
        raise ValueError('unregistered Level 3 operation')


def _recovery(model):
    if model.rows and model.rows[-1]['result'] == GENERATION_TRUNCATED and model.rows[-1]['usage'] == 'settled':
        return GENERATION_TRUNCATED
    if model.rejections and not model.rejections[-1]['dispatched'] and model.rejections[-1]['reason'] == 'context_generation_window':
        return 'context_generation_window'
    return None


def _fresh_segment_recovery(model):
    recover = _recovery(model)
    if recover or not model.rejections:
        return recover
    last = model.rejections[-1]
    if (last['reason'] == 'upstream_transport' or
            (last['reason'] == 'upstream_http' and last.get('http_status') in {408,429,500,502,503,504})):
        return 'network_failure'
    return None


def _budget_limited(model):
    """Distinguish local token-budget admission from a Pi/transport failure."""
    return bool(model.rejections and not model.rejections[-1].get('dispatched')
                and model.rejections[-1].get('reason') == 'token_reservation')


def explore(output, contract, worker, launcher, lease_factory, runtime, prompt, limits, *, node_ms, calls,
            reservations, research=None, holdout_review=None, bootstrap_measure=False, paired_timing=False, scoring=None,
            consecutive_failure_limit=3, seed_engine=None, resume=None, continuous=False, notify=None, weight_audit=None):
    """Run a fresh exploration. Live admission remains the invoking trusted CLI's responsibility."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = monotonic()
    deadline = None if limits.wall_ms is None else started + limits.wall_ms/1000
    need((node_ms is None) == (calls is None) == (deadline is None), 'inconsistent cumulative resource policy')
    need(not continuous or deadline is None and isinstance(scoring, dict) and
         scoring.get('version') in business.NATURAL_VERSIONS, 'continuous exploration requires metered natural EOS')
    need(seed_engine is None or scoring and bootstrap_measure and paired_timing, 'seed candidate requires paired engineering scoring')
    need(resume is None or seed_engine is not None, 'portable resume requires a seed engine')
    lineage = Lineage(output/'harness-lineage', runtime)
    lineage.initialize(Snapshot.make({}, EMPTY, runtime))
    spent_input = spent_output = spent_requests = 0
    rows, final, reason = [], None, 'unknown'
    with Journal(output/'trajectory.sqlite3', external_serialization=True) as journal:
        if weight_audit:
            journal.event('weights/policy', {'policy':weight_audit})
        if scoring:
            journal.event('business_scoring/policy', business.resolve_policy(scoring))
        if holdout_review is not None:
            metadata, approved_hash = holdout_review
            register_review(journal, metadata, approved_sha256=approved_hash)
        state = Run(journal, contract, lineage, worker, deadline, node_ms=node_ms, calls=calls, reservations=reservations, research=research,
                    require_measurement=bootstrap_measure, paired_timing=paired_timing, scoring=scoring,
                    continuous=continuous, notify=notify, weight_audit=weight_audit)
        e0 = contract.from_args({'files': {}})
        state.e0 = e0
        state.add_engine(e0)
        last, segment, failure = None, None, None
        try:
            need(state.develop('develop_build', e0)['status'] == 'built', 'E0 build failed')
            need(state.develop('develop_quality', e0)['status'] == 'valid', 'E0 quality failed')
            if bootstrap_measure:
                measured = state.develop('develop_measure', e0, timing_pair=0 if paired_timing else None)
                need(measured['status'] == 'observed', 'E0 timing failed')
                reference = state.receipts[measured['evidence']]
                compare_dev(reference, reference)  # Reject malformed metrics before opening Pi.
                state.reference = reference
                state._event('development_reference', engine=e0.digest, evidence=state.reference.evidence)
            initial = e0
            if seed_engine is not None and seed_engine.digest != e0.digest:
                state.add_engine(seed_engine, e0.digest)
                built = state.develop('develop_build', seed_engine)
                if built['status'] == 'built':
                    quality = state.develop('develop_quality', seed_engine)
                    if quality['status'] == 'valid':
                        measured = state.measure_pair(seed_engine)
                        if measured['status'] == 'observed':
                            initial = seed_engine
                state._event('seed_candidate_retested', seed=seed_engine.digest, starting_engine=initial.digest)
            wip_id = None
            if resume is not None:
                if initial.digest == seed_engine.digest:
                    snapshot = resume['harness']
                    current = lineage.active()
                    version = lineage.submit(MODE, dict(snapshot.items), snapshot.manifest['load'])
                    lineage.activate(MODE, version, current.identity, actor='controller_resume',
                                     request_id=resume['checkpoint'])
                    if resume['wip'] is not None:
                        wip_id = state.add_engine(resume['wip'], initial.digest)
                    if resume.get('asset_identities'):
                        state.assets.import_objects(resume['asset_root'], resume['asset_identities'])
                    if resume.get('research_identities') and isinstance(state.research, WebResearch):
                        state.research.import_pages(resume['research_root'], resume['research_identities'])
                    state.historical_facts = resume['recent_facts']
                    state._event('checkpoint_imported', checkpoint=resume['checkpoint'], engine=initial.digest,
                                 historical_facts=resume['recent_facts'])
                else:
                    state._event('checkpoint_import_refused', checkpoint=resume['checkpoint'])
            last = state.checkpoint(initial.digest, wip_id)
            segment = state.lifecycle.start(last)
        except (ValueError, RuntimeError, OSError) as exc:
            reason, failure = 'bootstrap_failed', type(exc).__name__
            state._event(reason, error=failure)
        previous_failure, consecutive_failures = None, 0
        historical_score, stalled_segments = float('-inf'), 0
        while segment is not None:
            # Trusted host marker only. Finish the current segment and close all leases normally.
            if continuous and weight_audit and (output/'pause-after-segment').is_file():
                reason = 'operator_pause_after_segment'
                state.lifecycle.stop(segment, cause=reason)
                break
            left = None if deadline is None else int((deadline-monotonic())*1000)
            remains = (None, None, None) if left is None else (
                limits.input-spent_input, limits.output-spent_output, limits.turns-spent_requests)
            if left is not None and (left < 1000 or min(remains) <= 0 or remains[1] < limits.per_turn_output):
                reason = 'resource_limit'
                state.lifecycle.stop(segment, cause=reason)
                break
            entry = state.lifecycle.entry(segment)
            current = lineage.get(entry.harness)
            draft = state.engines[state.drafts[entry.wip][1]] if entry.wip else None
            files, load = dict(current.items), current.manifest['load']
            parts = tuple(files[name].decode() for name in load['prompts']+load['memory'])
            lease = lease_factory(replace(limits, input=remains[0], output=remains[1], turns=remains[2], wall_ms=left), segment, parts)
            lease.deadline = deadline
            state.pending = None
            state.segment_checkpoint = None
            if continuous:
                state._event('segment_started', segment=segment, checkpoint=last, engine=entry.engine.engine, harness=entry.harness)
            entry_prompt = prompt + f'\n\nSegment {segment}. Verified Engine: {entry.engine.engine}. Active Harness: {entry.harness}. Use facts for current state.'
            gateway = None
            # Separate append-only stream; do not put exception stacks into Pi or its public trace.
            audit = RpcAudit(output/f'rpc-segment-{segment:03}.jsonl')
            def job_audit(**record):
                gateway._audit(**record)
            try:
                with state.handler(segment, entry, audit=job_audit) as dispatch:
                    gateway = Gateway(lease, dispatch, delivery_complete=lambda: state.pending is not None, audit=audit)
                    report = launcher(state.engines[entry.engine.engine], current, entry_prompt, gateway,
                                      seconds=None if left is None else left//1000, draft=draft)
            except KeyboardInterrupt:
                state._event('operator_interrupted', segment=segment, checkpoint=state.latest_checkpoint)
                state.lifecycle.stop(segment, cause='operator_interrupted')
                raise
            except (ValueError, RuntimeError, OSError, SubprocessError) as exc:
                failure = type(exc).__name__
                state._event('launcher_failed', segment=segment, error=failure)
                report = dict(launcher_failure=failure, provider_rows=lease.rows,
                              gateway_events=gateway.events if gateway is not None else [])
            if state.rpc_unsettled:
                # The callback can still own the journal. Do not publish a new
                # segment or write to the same journal from this thread. The outer
                # launcher settles conservatively from already durable receipts.
                (output/'runtime-quarantine.json').write_bytes(canonical(dict(
                    stop_cause='trusted_rpc_unsettled', segment=segment, checkpoint=last,
                    watchdog_fault=report.get('watchdog_fault'), launcher_failure=report.get('launcher_failure')))+b'\n')
                raise RuntimeError('trusted RPC did not drain; operator reconciliation required')
            spent_input += lease.spent_input
            spent_output += lease.spent_output
            spent_requests += len(lease.rows)
            row = dict(segment=segment, entry=asdict(entry), report=report, provider_rejections=lease.rejections,
                       cumulative=dict(input=spent_input, output=spent_output, requests=spent_requests))
            rows.append(row)
            trace_object = journal.put('level3_segment', row)
            if continuous:
                best = state.historical_best()
                state._event('segment_settled', segment=segment, harness=entry.harness, best=best,
                             model_usage=row['cumulative'], trace_object=trace_object, trace_sha256=digest(canonical(row)))
            resources_clean = (all(report.get(k) is True for k in ('isolated_state_removed','socket_removed','unchanged_install'))
                               and report.get('relay_settled') is not False and report.get('pi_reader_settled') is not False
                               and not report.get('timed_out') and not state.worker_invalid)
            watchdog = report.get('watchdog_fault')
            clean = resources_clean and not (watchdog or report.get('transport_fault'))
            terminated = clean and report.get('pi_exit') == 0 and not report.get('pi_errors')
            if continuous and terminated:
                previous_failure, consecutive_failures = None, 0
                operation, value = state.pending or ('continue', None)
                if operation == 'nominate':
                    proposed, checkpoint, score = value
                    proposal = state.lifecycle.nominate(segment, proposed, checkpoint, active_harness=lineage.active().identity,
                                  score=score, cold_verified=state.builds[proposed.engine].data.get('build_mode') == 'independent_cold')
                    state.audit_candidate('final_nomination', proposed, segment=segment)
                    state.stage_proposals.append(proposal)
                    state._event('stage_proposal', **proposal)
                else:
                    checkpoint = value or state.continuation_checkpoint(last)
                marker = best['weighted_score'] if best else float('-inf')
                if marker > historical_score:
                    stalled_segments, historical_score = 0, marker
                else:
                    stalled_segments += 1
                    if stalled_segments % 3 == 0:
                        state._event('stagnation_notice', segment=segment, stalled_segments=stalled_segments, best=best)
                        if notify:
                            notify('stagnation_notice', dict(segment=segment, stalled_segments=stalled_segments))
                segment = state.lifecycle.handoff(segment, checkpoint, active_harness=lineage.active().identity)
                last = checkpoint
                continue
            if state.pending and terminated:
                previous_failure, consecutive_failures = None, 0
                operation, value = state.pending
                if operation == 'deliver':
                    state.audit_candidate('final_nomination', value, segment=segment)
                    final = state.lifecycle.finish(segment, value, active_harness=lineage.active().identity,
                                                   pending_review=bool(weight_audit))
                    reason = 'agent_nomination_pending_review' if weight_audit else 'agent_delivery'
                    break
                segment = state.lifecycle.handoff(segment, value, active_harness=lineage.active().identity)
                last = value
                continue
            recover = ((_fresh_segment_recovery(lease) if deadline is None else _recovery(lease))
                        if clean and not state.pending else None)
            if watchdog:
                watchdog_cleanup = resources_clean and all(report.get(k) is True for k in ('relay_settled','pi_reader_settled'))
                state._event('guest_watchdog', segment=segment, fault=watchdog, cleanup_verified=watchdog_cleanup)
                if watchdog_cleanup and not state.pending and all(row.get('usage') == 'settled' for row in lease.rows):
                    recover = 'guest_failure'
            if deadline is None and recover in RECOVERABLE:
                # For a guest failure, model turns alone are not recovery progress.
                progressed = (state.latest_checkpoint != last if recover == 'guest_failure' else
                              any(r.get('result') == 'complete' for r in lease.rows))
                consecutive_failures = consecutive_failures + 1 if recover == previous_failure and not progressed else 1
                previous_failure = recover
                if consecutive_failures >= consecutive_failure_limit:
                    reason = 'repeated_failure_paused'
                    state._event(reason, cause=recover, consecutive=consecutive_failures)
                    state.lifecycle.stop(segment, cause=reason)
                    break
                if recover == 'network_failure':
                    delay = min(60, 5 * 2 ** (consecutive_failures - 1))
                    state._event('infrastructure_backoff', segment=segment, cause=recover, seconds=delay)
                    sleep(delay)
            if recover in RECOVERABLE:
                # Last explicit checkpoint is optional; the active Harness must be bound at recovery time.
                candidate = [event['checkpoint'] for event in state.events if event['operation'] == 'checkpoint_saved'][-1]
                cp = state.lifecycle.load(candidate)
                if cp.harness != lineage.active().identity:
                    candidate = state.lifecycle.save(replace(cp, harness=lineage.active().identity))
                    state.audit_candidate('checkpoint', cp.engine, checkpoint=candidate, recovery=True)
                try:
                    segment = state.lifecycle.handoff(segment, candidate, active_harness=lineage.active().identity,
                                                      abnormal_cause=None if deadline is None and recover == 'context_generation_window' else recover)
                    last = candidate
                    continue
                except ValueError as e:
                    if str(e) != 'recovery cap reached':
                        raise
                    reason = 'recovery_cap_reached'
            else:
                if state.worker_invalid:
                    reason = 'worker_uncertain'
                elif report.get('relay_settled') is False:
                    reason = 'trusted_rpc_unsettled'
                elif lease.rows and lease.rows[-1].get('usage') == 'usage_uncertain':
                    reason = 'usage_uncertain'
                elif report.get('timed_out'):
                    reason = 'wall_timeout'
                elif _budget_limited(lease):
                    reason = 'resource_limit'
                else:
                    reason = 'no_explicit_delivery' if terminated else 'runtime_failure'
            state.lifecycle.stop(segment, cause=reason)
            break
        if continuous:
            state._event('exploration_paused', cause=reason, checkpoint=state.latest_checkpoint)
            if notify:
                notify('exploration_paused', dict(cause=reason, checkpoint=state.latest_checkpoint))
        result = dict(status=('pending_weight_review' if weight_audit else 'delivered') if final else 'incomplete', stop_cause=reason, final=final, segments=rows,
                      jobs=state.jobs, events=state.events, last_checkpoint=last, node_ms=sum(job.get('node_ms', job['reserved_ms']) for job in state.jobs),
                      development_reference=state.reference.evidence if state.reference else None,
                      wallet=dict(input=spent_input, output=spent_output, requests=spent_requests), failure=failure,
                      wall_ms=max(0, round((monotonic()-started)*1000)))
        if weight_audit:
            result.update(weight_audit=weight_audit, gain_status=weights.PROVISIONAL)
        if scoring:
            result['business_scoring'] = state.score_policy
            result['best_eligible'] = state.business_best()
            result['chosen_business_score'] = state.business_summary(final['chosen']['engine']) if final else None
            if continuous:
                result['stage_proposals'] = state.stage_proposals
                result['historical_best'] = state.historical_best()
        (output/'result.json').write_bytes(canonical(result)+b'\n')
        return result
