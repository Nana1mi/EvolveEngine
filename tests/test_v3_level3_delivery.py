import json
from math import nextafter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evolveengine.journal import Journal
from evolveengine.v3_harness import Lineage, Snapshot, canonical, digest
from evolveengine.v3_level3_delivery import export_delivery
from evolveengine.v3_level3_lifecycle import ROOT
from evolveengine.v3_level3_run import compare_dev, WorkerResult
from evolveengine.v3_level3_source import SourceContract


class DeliveryExportTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / 'bundle'
        run = self.root / 'run'
        run.mkdir()
        path = 'python/sglang/srt/scheduler.py'
        self.contract = SourceContract({path: b'x = 1\n'})
        source = self.contract.from_args({'files': {path: 'x = 2\n'}})
        engine = source.digest
        e0 = self.contract.from_args({'files': {}}).digest
        build, passed, refreshed = '1' * 64, '2' * 64, '3' * 64
        runtime = dict(pi_image='a' * 64, package_lock_sha='b' * 64, runner_sha='c' * 64,
                       base_config_sha='d' * 64, node_version='24.13.0', pi_version='1.5.2')
        layout = dict(prompts=[], skills=[], extensions=[], workflows=[], memory=['memory/notes.md'])
        lineage = Lineage(run / 'harness-lineage', runtime)
        self.harness = lineage.initialize(Snapshot.make({'memory/notes.md': b'notes\n'}, layout, runtime))
        self.chosen = dict(engine=engine, build=build, quality=refreshed)
        qualification = dict(self.chosen, quality=passed)
        final = dict(kind='delivered', chosen=self.chosen, active_harness=self.harness, segment=1)
        measure = ('develop_measure', 'observed', 1)
        baselines, candidates = (dict(scenes={scene: {'dev_score': score} for scene in ('W1', 'W2', 'W3', 'W4')},
                                     measurement_inputs_sha256='8' * 64, timed_outputs_sha256='9' * 64, measurement_pair=4)
                                 for score in (100, 150))
        e0_result = WorkerResult(e0, *measure, '4' * 64, '5' * 64, baselines)
        candidate = WorkerResult(engine, *measure, build, '6' * 64, candidates)
        comparison = compare_dev(e0_result, candidate)
        comparison['equal_scene_geomean'] = nextafter(comparison['equal_scene_geomean'], 2)
        with Journal(run / 'trajectory.sqlite3') as journal:
            object_id = journal.put('level3_engine', self.contract.to_args(source))
            journal.event(ROOT + 'delivery', final)
            journal.event(ROOT + 'quality/' + refreshed, self.chosen)
            journal.event(ROOT + 'quality/' + passed, qualification)
            journal.event(ROOT + 'measure/' + candidate.evidence, dict(qualification, receipt=candidate.evidence))
        events = [dict(operation='engine_registered', engine=engine, source_object=object_id)]
        jobs = [dict(operation='develop_build', status='built', engine=engine, evidence=build),
                dict(operation='develop_quality', status='valid', engine=engine, build=build, evidence=refreshed),
                vars(e0_result), dict(vars(candidate), performance=comparison)]
        self.result = dict(status='delivered', stop_cause='agent_delivery', final=final, jobs=jobs, events=events,
                           node_ms=5, development_reference=e0_result.evidence)
        self.trajectory = dict(points=[dict(engine=engine, build=build, development=comparison, measurement_pair=4,
                                            measurement_inputs_sha256='8' * 64, timed_outputs_sha256='9' * 64)],
                               charged_node_ms=5, reference=e0_result.evidence)
        self.write('run/result.json', self.result)
        self.write('performance-trajectory.json', self.trajectory)
        self.write('registration.json', {'runtime': runtime})
        self.write('settlement.json', dict(status='delivered', last_phase='worker_close', node_ms=5,
                                           worker_cost={'unsettled_node_reservation_ms': 0}))

    def write(self, name, data):
        (self.root / name).write_bytes(canonical(data) + b'\n')

    def test_exports_selected_source_harness_and_pair_after_quality_recheck(self):
        result = export_delivery(self.root, self.output, self.contract)
        source = (self.output / 'engine-source.json').read_bytes()
        self.assertEqual(result['source_sha256'], digest(source))
        self.assertEqual(self.contract.from_args(json.loads(source)).digest, self.chosen['engine'])
        self.assertAlmostEqual(result['paired_development'][0]['geomean'], 1.5)
        self.assertEqual(result['active_harness'], self.harness)
        self.assertEqual((self.output / 'harness/files/memory/notes.md').read_bytes(), b'notes\n')
        with self.assertRaises(FileExistsError):
            export_delivery(self.root, self.output, self.contract)

    def test_does_not_publish_if_paired_output_evidence_differs(self):
        self.trajectory['points'][0]['timed_outputs_sha256'] = '0' * 64
        self.write('performance-trajectory.json', self.trajectory)
        with self.assertRaisesRegex(ValueError, 'Paired input or greedy output evidence'):
            export_delivery(self.root, self.output, self.contract)
        self.assertFalse(self.output.exists())

    def test_does_not_publish_a_changed_score_even_when_both_json_files_agree(self):
        self.trajectory['points'][0]['development']['equal_scene_geomean'] = 1.51
        self.write('performance-trajectory.json', self.trajectory)
        self.write('run/result.json', self.result)
        with self.assertRaisesRegex(ValueError, 'Development comparison does not replay'):
            export_delivery(self.root, self.output, self.contract)
        self.assertFalse(self.output.exists())

    def test_does_not_treat_incomplete_run_as_delivery(self):
        self.result['status'] = 'incomplete'
        self.write('run/result.json', self.result)
        with self.assertRaisesRegex(ValueError, 'settled explicit Agent delivery'):
            export_delivery(self.root, self.output, self.contract)
        self.assertFalse(self.output.exists())
