"""v2.2 RQ-A only: a frozen policy, held-out H0 mothers and a detached blind judge."""
from dataclasses import asdict
import math
from statistics import mean

from .controller import Controller, HARNESS_REVISION, Lane
from .engine import gate
from .experiment import _remaining, final_comparison, open_sealed_judge
from .journal import InsufficientBudget
from .ranking import policy_rules
from .reports import paired_interval
from .schema import Checkpoint, Cost, digest


SEALED = 'evolveengine.sealed-rqa.v1'
REPORT = 'evolveengine.rqa.v1'
PERCENT = (50, 75)


def build_rqa_starts(controller, lane, engine, steps):
    """Sample completed H0 step events by elapsed wall time, never by outcome or step index."""
    j = controller.journal
    initial = controller.checkpoint(engine, lane.path)
    sealed = controller.run(lane, initial, steps, track_completions=True)  # H0, no updater or W
    history = [(0, lane.path + '/run/manifest', asdict(initial))]
    for i in range(sealed['steps']):
        key = f'{lane.path}/step-{i}/finished'
        row = j.get_event(key)
        if row['completed_wall_ms'] < history[-1][0]:
            raise ValueError('Mother wall clock moved backwards across completed checkpoints')
        history.append((row['completed_wall_ms'], key, row['checkpoint']))
    points = []
    for percent in PERCENT:
        boundary = lane.budget.wall_ms * percent // 100
        elapsed, source, state = next(row for row in reversed(history) if row[0] <= boundary)
        points.append({'percent': percent, 'wall_ms': boundary, 'completed_wall_ms': elapsed,
                       'source_event': source, 'checkpoint': state})
    j.event(lane.path + '/rq-a-starts', {'points': points})
    return points


def seal_rqa(journal, backend, partitions, protocol, nodes, replicate, environment, frozen_policy):
    """Seal both forks before Q/T is accessible. The caller supplies H*; this does not train it."""
    if type(replicate) is not int or replicate < 0 or not nodes:
        raise ValueError('Expected an independent replicate ID and a node allocation')
    policy_rules(frozen_policy)
    partitions = partitions.development_view()
    if not partitions.development_commitment or not partitions.final_commitment:
        raise ValueError('RQ-A requires separate development and final commitments')
    c = Controller(journal, backend, partitions, environment, epsilon=environment.get('noise_epsilon', 0.002))
    root = f'rq-a/replicate-{replicate}'
    node = nodes[replicate % len(nodes)]
    mother = Lane(root + '/mother', 'research_overhead', node, protocol.bank)
    forks = Lane(root + '/forks', 'research_overhead', node, protocol.evaluation)
    engine = backend.initial_engine()
    h_star = journal.put('policy', frozen_policy)
    manifest = {'endpoint': REPORT, 'harness_revision': HARNESS_REVISION, 'environment': c.environment,
                'replicate': replicate, 'mother_lineage': mother.path, 'node': node,
                'initial_engine': journal.put('engine', engine), 'initial_policy': c.baseline,
                'frozen_policy': h_star, 'simulated': environment.get('simulated', False),
                'protocol': asdict(protocol), 'partitions': partitions.public_manifest()}
    journal.event(root + '/manifest', manifest)  # Changing H* on resume is forbidden.
    saved = journal.get_event(root + '/sealed')
    if saved is not None:
        return saved
    starts = build_rqa_starts(c, mother, engine, protocol.bank_steps)
    journal.begin_scope(forks.path)
    groups, engines = [], {}
    for start in starts:
        checkpoint = Checkpoint(**start['checkpoint'])
        key = f"at-{start['percent']}"
        ref = journal.get('engine', checkpoint.engine_digest)
        engines[checkpoint.engine_digest] = ref
        group = {'start': start}
        try:
            admitted = c._operate(forks, key + '/admit', 'quality', {'engine': ref, 'role': 'admission',
                'block': partitions.block('admission', 'Q_admit', f'{forks.path}/{key}/admit', 32)})
            if not gate(admitted):
                group['status'] = 'invalid_admission'
            else:
                pair = c.paired_steps(forks, key, checkpoint, c.baseline, h_star, protocol.causal, protocol.fork_steps)
                group.update(status='sealed', submissions=pair)
                for delivery in pair.values():
                    address = delivery['checkpoint']['engine_digest']
                    engines[address] = journal.get('engine', address)
        except InsufficientBudget:
            group['status'] = 'fork_budget_exhausted'
        groups.append(group)
    spent = journal.spent(forks.path)
    payload = {'schema': SEALED, 'environment': environment, 'run_manifest': manifest,
               'engines': engines, 'points': groups, 'mother_cost': journal.spent(mother.path).record(),
               'fork_cost': spent.record(), 'remaining_evaluation_budget': _remaining(
                   protocol.evaluation, spent, journal.wall_elapsed(forks.path, spent)).record()}
    bundle = {'digest': digest(payload), 'payload': payload}
    journal.event(root + '/sealed', bundle)
    return bundle


def evaluate_rqa(journal, backend, partitions, protocol, bundle, expected_digest):
    payload, c = open_sealed_judge(journal, backend, partitions, protocol, bundle, expected_digest, schema=SEALED)
    m = payload['run_manifest']
    _check_points(m, [g['start'] for g in payload['points']])
    lane = Lane(f"rq-a/replicate-{m['replicate']}/judge", 'research_overhead', m['node'],
                Cost(**payload['remaining_evaluation_budget']))
    journal.event(lane.path + '/input', {'submission_digest': expected_digest})
    journal.begin_scope(lane.path)
    points = []
    for group in payload['points']:
        start = group['start']
        key = f"at-{start['percent']}"
        name = f'{lane.path}/{key}/result'
        blind = journal.get_event(name)
        if blind is None:
            blind = {'status': group['status']}
            if group['status'] == 'sealed':
                ref = Checkpoint(**start['checkpoint'])
                # A measured start is needed even when both other deliveries fall back. On the
                # same T block use actual delivered token/s, not two separately normalized gains.
                submitted = {**group['submissions'], 'start': {'checkpoint': asdict(ref)}}
                try:
                    blind = final_comparison(c, lane, key, ref, submitted, protocol, reuse_identical=True)
                except InsufficientBudget:
                    blind = {'status': 'evaluation_budget_exhausted'}
            journal.event(name, blind)
        points.append({'start': start, 'blind': blind})
    return {'schema': REPORT, 'run_manifest': m, 'submission_digest': expected_digest, 'points': points,
            'mother_cost': payload['mother_cost'], 'fork_cost': payload['fork_cost'],
            'judge_cost': journal.spent(lane.path).record()}


def _blind_contrast(blind, block_count):
    if blind['status'] != 'ok':
        return None  # Missing evidence is not a legal identical-engine zero.
    engines = blind['engines']

    def rates(label):
        row = engines[label]
        if row['status'] in ('fallback_quality', 'fallback_performance'):
            row = engines['start']
        if row['status'] != 'valid':
            raise ValueError('No registered deployable throughput for this arm')
        values = [block['candidate'] for block in row['tokens_per_second']]
        if (len(values) != block_count or not values or any(type(x) not in (int, float)
                or not math.isfinite(x) or x <= 0 for x in values)):
            raise ValueError('Invalid or incomplete blind throughput blocks')
        return values

    return mean(math.log(b) - math.log(a) for a, b in zip(rates('parent'), rates('proposed')))


def _check_points(manifest, starts):
    identity = manifest['mother_lineage']
    if (manifest['endpoint'] != REPORT or identity != f"rq-a/replicate-{manifest['replicate']}/mother"
            or [p['percent'] for p in starts] != list(PERCENT)):
        raise ValueError('Need the two predeclared wall-clock points of one held-out H0 mother')
    for start in starts:
        state = start['checkpoint']
        if (state['lineage'] != identity or state['policy_digest'] != manifest['initial_policy']
                or state['environment_digest'] != manifest['environment']
                or start['wall_ms'] != manifest['protocol']['bank']['wall_ms'] * start['percent'] // 100
                or not 0 <= start['completed_wall_ms'] <= start['wall_ms']):
            raise ValueError('RQ-A point is not the registered H0 mother/time')


def summarize_rqa(reports):
    """First average the two log contrasts within their mother, then form the paired t interval."""
    signature = None
    seen, values, incomplete = set(), [], []
    simulated = None
    for report in reports:
        if report.get('schema') != REPORT or report['run_manifest'].get('endpoint') != REPORT:
            raise ValueError('RQ-A cannot aggregate a v1 or auxiliary E0 report')
        m = report['run_manifest']
        identity = m['mother_lineage']
        if identity in seen:
            raise ValueError('Duplicate or inconsistent held-out mother lineage')
        seen.add(identity)
        simulated = m['simulated']
        part = m['partitions']
        sig = digest([m[k] for k in ('harness_revision', 'environment', 'initial_engine', 'initial_policy',
                                      'frozen_policy', 'simulated', 'protocol')] +
                     [part[k] for k in ('generator', 'lengths', 'token_set_digest', 'special_prefix')])
        if signature is not None and sig != signature:
            raise ValueError('Cannot mix frozen policies, protocols, environments or harness revisions')
        signature = sig
        points = report['points']
        _check_points(m, [p['start'] for p in points])
        ds = [_blind_contrast(p['blind'], m['protocol']['final_blocks']) for p in points]
        if None in ds:
            incomplete.append({'lineage': identity, 'statuses': [p['blind']['status'] for p in points]})
        else:
            values.append(mean(ds))
    result = paired_interval(values)
    positive = 'ci95' in result and result['ci95'][0] > 0
    return {**result, 'lineage_count': len(seen), 'incomplete_lineages': incomplete, 'simulated': simulated,
            'positive_effect': positive, 'practical_reference': 0.03,
            'at_least_3pct_point_estimate': result['geometric_advantage'] >= 0.03 if values else None,
            'confirmatory_pass': simulated is False and not incomplete and positive}
