import os
from pathlib import Path
from unittest import TestCase,skipUnless
from experiments.spark26_gguf.level3.native_controls import (
    ENGINE_GGUF, FAULT, KERNEL_WITNESS, MMQ, MMVQ, STORE,
    ROOT, SourceContract, make,
)


@skipUnless(os.environ.get('EVOLVEENGINE_NATIVE_ARCHIVE'),'requires the pinned SGLang archive')
class NativeControlsTests(TestCase):
    def test_actual_e0_controls_are_distinct_and_inputs_not_rewritten(self):
        c = SourceContract.acquire_spark26_e0(Path(os.environ['EVOLVEENGINE_NATIVE_ARCHIVE']),ROOT)
        v = make(c)
        self.assertEqual({k:x.digest for k,x in v.items()},{k:x.digest for k,x in make(c).items()})
        for role,path in (('native_positive',MMVQ),('native_compute_negative',MMQ),('residency_negative',ENGINE_GGUF)):
            self.assertEqual(set(dict(v[role].edits)),{path})
            self.assertNotEqual(v[role].digest,v['e0'].digest)
        pos = dict(v['native_positive'].edits)[MMVQ]
        self.assertEqual(pos.replace(KERNEL_WITNESS.encode(),b'mul_mat_vec_q'),c.base[MMVQ])
        neg = dict(v['native_compute_negative'].edits)[MMQ].decode()
        self.assertEqual(neg.count(FAULT),1)
        self.assertEqual(neg.removeprefix('#include <type_traits>\n').replace(FAULT,STORE).encode(),c.base[MMQ])
        self.assertIn(b'layer._fixture_bf16_weight_copy = dequantize_gguf_weight',dict(v['residency_negative'].edits)[ENGINE_GGUF])

    def test_legal_variant_is_batch_selected_and_keeps_both_numerical_paths(self):
        import json
        from experiments.spark26_gguf.level3 import legal_variant_control as legal
        from evolveengine.v3_level3_startup import PATH
        contract = SourceContract.acquire_spark26_e0(Path(os.environ['EVOLVEENGINE_NATIVE_ARCHIVE']), ROOT)
        first = legal.candidate(contract)
        self.assertEqual(first.digest, legal.candidate(contract).digest)
        edits = dict(first.edits)
        self.assertEqual(set(edits), {MMVQ, PATH})
        self.assertEqual(json.loads(edits[PATH]), legal.KNOBS)
        self.assertEqual(edits[MMVQ].count(b'if (nvecs == 1)'), 1)
        self.assertEqual(edits[MMVQ].count(b'Q8_1_MMVQ, vec_dot_q8_0_q8_1, true>'), 1)
        self.assertEqual(edits[MMVQ].count(legal.LAUNCH.encode()), 1)
        self.assertIn(b'ascending ? bit : (WARP_SIZE / (2 * bit))', edits[MMVQ])


class ReductionControlTests(TestCase):
    def test_butterflies_use_each_lane_once_but_can_round_differently(self):
        import struct
        def reduce(values, masks, add):
            for mask in masks:
                values=[add(x,values[i ^ mask]) for i,x in enumerate(values)]
            return values[0]
        def add32(x,y): return struct.unpack('f',struct.pack('f',x+y))[0]
        up=[1,2,4,8,16]; down=up[::-1]
        for masks in (up,down):
            self.assertCountEqual(reduce([[i] for i in range(32)], masks, list.__add__), range(32))
        x=[1.e8,1.,-1.e8,1.] + [0.]*28
        self.assertNotEqual(reduce(x,up,add32),reduce(x,down,add32))


class SupplementalPositiveReviewTests(TestCase):
    def test_cross_checks_result_fixture_plan_and_candidate_receipt(self):
        import sys
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from evolveengine.v3_level3_collect import write
        from evolveengine.v3_harness import digest
        from experiments.spark26_gguf.level3 import legal_variant_control as legal
        with patch.object(sys, 'path', [str(Path(legal.__file__).parent), *sys.path]):
            from draft_development_admission import verify_extra
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); primary = root/'primary'; controls = primary/'controls'; extra = root/'extra'
            controls.mkdir(parents=True); extra.mkdir()
            bundle = b'synthetic primary bundle'
            (primary/'bundle.tgz').write_bytes(bundle)
            prereg = dict(primary_bundle=digest(bundle), engine='e'*64,
                          driver=digest(Path(legal.__file__).read_bytes()),
                          expected=['built','built','valid'])
            prereg_path = root/'prereg.json'
            prereg_ref = (prereg_path,write(prereg_path,prereg))
            def check(path, pin, acceptance):
                return verify_extra(primary, path, pin, acceptance, prereg_ref=prereg_ref)
            report = {key: str(i)*64 for i,key in enumerate(('registration','trajectory','graph','noise'),1)}
            e0 = 'a'*64; variant = prereg['engine']; base_proof = 'b'*64; candidate_proof = 'c'*64
            acceptance = write(controls/'acceptance.json', dict(report, identities={'base':e0}))
            plan = write(extra/'plan.json',dict(report, driver=prereg['driver'], acceptance=acceptance,
                engine=variant, runtime=legal.KNOBS, expected=prereg['expected'], scope='quality_positive_not_performance',
                retry=False,holdout=False))
            args=['--context-length','8192','--stream-interval','1','--max-running-requests','6',
                  '--cuda-graph-bs-decode','1','2','3','4','5','6']
            base = dict(engine=e0, status='built', operation='develop_build', node_ms=42, evidence=base_proof, build=None,data={})
            candidate = dict(base, engine=variant, evidence=candidate_proof, data={'runtime':{'argv':args}})
            quality = dict(engine=variant, status='valid', operation='develop_quality', node_ms=3000, evidence='d'*64,
                           build=candidate_proof, data=dict(weight_observation={'scope':'memory'},
                               trajectory_screen={'status':'valid','calibration':report['trajectory']},
                               case_outcomes=dict.fromkeys(('numeric_32','natural_eos_4','paired_cancel_numeric','autoregressive_full_vocab'),'pass')))
            results = {label:write(extra/(label+'.json'),dict(result=row,verified=True))
                       for label,row in zip(('base','candidate','quality'),(base,candidate,quality),strict=True)}
            p = extra/'result.json'
            report = dict(plan=plan, engine=variant, receipts=results, verdict='valid', holdout=False, paid_calls=0,
                          scope='candidate_quality_not_speed_or_per_shape_kernel_trace')
            pin = write(p,report)
            self.assertEqual(check(p,pin,acceptance)['engine'],variant)
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                check(p,'9'*64,acceptance)
            for variant_dir, label, change, message in (
                ('bad_build','quality',dict(quality,build=base_proof),'approved candidate build'),
                ('bad_argv','candidate',dict(candidate,data={'runtime':{'argv':[*args,'--disable-radix-cache']}}),'startup policy'),
                ('missing_case','quality',dict(quality,data=dict(quality['data'],case_outcomes={'numeric_32':'pass'})),
                 'approved candidate build'),
            ):
                alt = root/variant_dir; alt.mkdir()
                (alt/'plan.json').write_bytes((extra/'plan.json').read_bytes())
                for source in ('base','candidate','quality'):
                    if source != label:
                        (alt/(source+'.json')).write_bytes((extra/(source+'.json')).read_bytes())
                ref = write(alt/(label+'.json'),dict(result=change,verified=True))
                pin = write(alt/'result.json',dict(report,receipts=results|{label:ref}))
                with self.assertRaisesRegex(ValueError,message):
                    check(alt/'result.json',pin,acceptance)
            (primary/'bundle.tgz').write_bytes(b'other source bundle')
            with self.assertRaisesRegex(ValueError,'wrong primary source bundle'):
                check(p, '9'*64, acceptance)
