from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evolveengine.deepseek import Provider
from evolveengine.pi import PiAgent
from evolveengine.schema import Cost
from evolveengine.workspace import Repository
from experiments.p1.trial import LIMIT, TrialBudget, public_check, run_trial

SRC = 'python/minisgl/demo.py'
CARD = '只做公开玩具题。'.encode()
PROVIDER = Provider('http://127.0.0.1:1','fixture','FIXTURE_NEVER_USE','fixture',256)


class Repo:
    editable = ('python/minisgl',)
    editable_snapshot = staticmethod(Repository.editable_snapshot)

    @contextmanager
    def checkout(self,parent):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); fn=p/SRC;fn.parent.mkdir(parents=True);fn.write_text('value = 1\n');yield p


def trial(pi,arm='short',**overrides):
    kw=dict(arm=arm,card_bytes=CARD,card_sha256=sha256(CARD).hexdigest(),scope=(SRC,));kw.update(overrides)
    return run_trial(pi,Repo(),{},**kw)


class Scripted:
    provider=PROVIDER
    def __init__(self,steps):self.steps=iter(steps);self.calls=[]
    def session(self,path,prompt,broker,**scope):
        self.calls.append((path,prompt,id(broker),broker.deadline));step=next(self.steps)
        if isinstance(step,Exception):raise step
        status,edit=step
        if edit is not None:(path/SRC).write_text(edit)
        broker.rows.append({'usage':'stub'}); broker.spent+=Cost(input_tokens=20,output_tokens=5)
        return {'status':status,'reason':'toy','intent':'toy'},[]


class P1Tests(unittest.TestCase):
    def test_correct_once_same_tree_cap_and_card(self):
        pi=Scripted([('implemented',None),('implemented','value=2')]);r=trial(pi)
        self.assertEqual([x['public_feedback']['code'] for x in r['calls']],['empty_patch','ok']);self.assertEqual(r['edits'],{SRC:'value=2'});self.assertEqual(r['cost']['input_tokens'],40);self.assertIsNone(r['independent_verdict'])
        a,b=pi.calls;self.assertEqual((a[0],a[2:]),(b[0],b[2:]));self.assertIn(CARD.decode(),a[1]);self.assertTrue(b[1].startswith(a[1]));self.assertFalse(a[0].exists())
    def test_single_computes_check_but_does_not_repair(self):
        pi=Scripted([('implemented',None)]);r=trial(pi,'single');self.assertEqual(len(pi.calls),1);self.assertEqual(r['calls'][0]['public_feedback']['code'],'empty_patch')
    def test_decline_and_pass_get_no_extra_turn(self):
        for status,edit in [('infeasible',None),('implemented','value=3')]:
            pi=Scripted([(status,edit)]);self.assertEqual(trial(pi)['state'],'submitted');self.assertEqual(len(pi.calls),1)
    def test_second_error_never_retried(self):
        pi=Scripted([('implemented','value=('),('implemented','value=(')]);r=trial(pi);self.assertEqual(len(r['calls']),2);self.assertEqual(r['calls'][-1]['public_feedback']['code'],'syntax_error')
    def test_hash_mismatch_before_session(self):
        pi=Scripted([])
        with self.assertRaises(ValueError):trial(pi,card_bytes=b'wrong')
        self.assertFalse(pi.calls)
    def test_provider_deadline_not_repaired(self):
        r=trial(Scripted([('implemented','value=('),subprocess.TimeoutExpired('mock',1)]));self.assertEqual(r['state'],'failed');self.assertFalse(r['edits']);self.assertEqual(r['cost']['output_tokens'],5)
    def test_static_check_does_not_run_source_or_reflect_its_text(self):
        receipt={'status':'implemented','intent':'x','reason':'none'};self.assertEqual(public_check(receipt,{SRC:'raise Exception("not executed")'})['code'],'ok');out=public_check(receipt,{SRC:'secret_value = ('});self.assertEqual(set(out),{'code','path','line'});self.assertNotIn('secret_value',str(out))
    def test_directional_caps_cannot_borrow_or_reset(self):
        b=TrialBudget(PROVIDER,LIMIT);b.spent=Cost(input_tokens=13400,output_tokens=3990);self.assertEqual(b.turn_output_ceiling(100),10);self.assertEqual(b.turn_output_ceiling(101),0);b.spent=Cost(output_tokens=4000);self.assertEqual(b.turn_output_ceiling(100),0)
    @unittest.skipUnless('EVOLVE_PI_TEST_COMMAND' in os.environ,'optional pinned Pi CLI')
    def test_real_pi_two_guarded_calls_one_mock_provider_budget(self):
        pi=PiAgent(json.loads(os.environ['EVOLVE_PI_TEST_COMMAND']),'1.5.2',PROVIDER);seen=[]
        def response(_url,*,payload,timeout):
            req=json.loads(payload);seen.append(req);repair='One correction remains' in json.dumps(req['messages']);used=any(x.get('role')=='tool' for x in req['messages']); tag='stop';delta={'role':'assistant','content':json.dumps({'status':'implemented','intent':'toy','reason':'toy'})}
            if repair and not used:
                delta={'role':'assistant','tool_calls':[{'index':0,'id':'t1','type':'function','function':{'name':'edit','arguments':json.dumps({'path':SRC,'edits':[{'oldText':'value = 1','newText':'value = 2'}]})}}]};tag='tool_calls'
            common={'id':'stub','object':'chat.completion.chunk','model':'fixture','created':1,'system_fingerprint':'fixture'};items=[{**common,'choices':[{'index':0,'delta':delta,'finish_reason':None}]},{**common,'choices':[{'index':0,'delta':{},'finish_reason':tag}]},{**common,'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':5,'total_tokens':25}}];return io.BytesIO(b''.join(b'data: '+json.dumps(x).encode()+b'\n\n' for x in items)+b'data: [DONE]\n\n')
        with patch.object(Provider,'request',side_effect=response):r=trial(pi)
        self.assertEqual(r['state'],'submitted',r['error']);self.assertEqual([x['public_feedback']['code'] for x in r['calls']],['empty_patch','ok']);self.assertEqual(r['edits'][SRC].strip(),'value = 2');self.assertEqual(r['cost']['input_tokens'],60);self.assertEqual(r['cost']['output_tokens'],15);self.assertEqual(len(seen),3)
