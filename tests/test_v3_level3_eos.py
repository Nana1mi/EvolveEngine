from hashlib import sha256
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from evolveengine.v3_level3_eos import CASES, grade, semantic_stop


class SemanticEosTests(TestCase):
    def grade(self, change=lambda rows: None, *, numeric='within_dev_screen', mixed=True):
        rows = []
        for index, case in enumerate(CASES):
            stop = dict(type='stop',matched=1)
            rows += [dict(kind=case,ids=[2,3,1],finish=stop,cap=48,input_sha=sha256(b'[8, 9]').hexdigest()),
                     dict(kind='natural_stream',case=case,prompt_ids=[8,9],
                          stream=dict(input=2,input_sha256=sha256(b'[8,9]').hexdigest(),observed_ids=[4,1],observed_finish=stop,sha256=sha256(b'[4,1]').hexdigest(),
                                      output=2,rid=f'{index:032x}',stream_complete=True,output_cap=48,finish_reason='stop'))]
        change(rows)
        legacy = dict(status='violation',max_tv=.0001,screen=dict(status=numeric),eos_passed=0,mixed_passed=mixed)
        with patch.dict('sys.modules',audit_worker=SimpleNamespace(quality=lambda *a:legacy)):
            return grade(rows,None,None)

    def test_different_answers_and_lengths_are_not_semantic_failures(self):
        result = self.grade()
        self.assertEqual((result['status'], result['eos_passed'], result['legacy_exact_matches_diagnostic']),('valid',4,0))
        self.assertEqual(self.grade(numeric='outside')['status'],'violation')
        self.assertEqual(self.grade(mixed=False)['status'],'violation')

    def test_premature_stop_missing_eos_and_hidden_tail_do_not_pass(self):
        for ids, finish in [([1],dict(type='stop',matched=1)),([2,3],dict(type='stop',matched=1)),
                            ([2,1,3,1],dict(type='stop',matched=1)),([2,1],dict(type='length')),
                            ([True,1],dict(type='stop',matched=1))]:
            with self.subTest(ids=ids,finish=finish):
                self.assertFalse(semantic_stop(ids,finish))
        self.assertEqual(self.grade(lambda rows: rows[0].update(ids=[1]))['status'],'violation')

    def test_missing_binding_is_not_a_verdict(self):
        for mutate in [lambda r:r.pop(),lambda r:r[1].update(prompt_ids=[7,7]),
                       lambda r:r[1]['stream'].update(sha256='0'*64),
                       lambda r:r[3]['stream'].update(rid=r[1]['stream']['rid'])]:
            with self.assertRaises(ValueError):
                self.grade(mutate)
