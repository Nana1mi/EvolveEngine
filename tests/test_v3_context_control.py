import json
import os
from pathlib import Path
import unittest

from evolveengine.v3_context_control import MARGIN_MIN, OfflineContextLease, PublicPromptCounter
from evolveengine.v3_provider import LeaseError, Limits, canonical


def request(messages=None):
    return dict(model='deepseek-flash',stream=True,thinking={'type':'enabled'},reasoning_effort='max',max_tokens=8192,
        messages=messages if messages is not None else [dict(role='system',content='toy system'),dict(role='user',content='toy task')])


def mock_stop():
    def block(delta=None,end=None,usage=None):
        data=dict(model='deepseek-flash',choices=[dict(index=0,delta=delta or {},finish_reason=end)])
        if usage:data['usage']=usage
        return b'data: '+canonical(data)+b'\n\n'
    return block(dict(reasoning_content='toy-thinking',content='done'))+block(end='stop',usage=dict(prompt_tokens=3,completion_tokens=2,total_tokens=5))+b'data: [DONE]\n\n'


class FakeCounter:
    def reservation(self,_):return dict(rendered_tokens=10,margin_tokens=MARGIN_MIN,local_candidate_bound=10+MARGIN_MIN)


class ContextControl(unittest.TestCase):
    def test_candidate_has_no_live_transport_and_replays_non_tool_reasoning(self):
        lv=Limits(5000,24000,4,10000,8192)
        with self.assertRaises(TypeError):OfflineContextLease(lv,FakeCounter(),responses=[],transport=lambda *_:None)
        l=OfflineContextLease(lv,FakeCounter(),responses=[mock_stop(),mock_stop()]);l.exchange(canonical(request()))
        base=request()['messages'];a=dict(role='assistant',content='done');u=dict(role='user',content='next')
        for invalid in (a,dict(a,reasoning_content=''),dict(a,reasoning_content='changed')):
            with self.assertRaises(LeaseError):l.exchange(canonical(request([*base,invalid,u])))
        self.assertEqual(len(l.mock_requests),1)
        self.assertEqual([r['reason'] for r in l.rejections],['reasoning_replay']*3)
        l.exchange(canonical(request([*base,dict(a,reasoning_content='toy-thinking'),u])))
        self.assertEqual(len(l.mock_requests),2)
        self.assertNotIn('toy-thinking',json.dumps([l.rows,l.rejections]))

    def test_no_counter_budget_fallback(self):
        class BrokenCounter:
            def reservation(self,_):raise ValueError('private fixture detail')
        l=OfflineContextLease(Limits(5000,24000,4,10000,8192),BrokenCounter(),responses=[mock_stop()])
        with self.assertRaises(LeaseError):l.exchange(canonical(request()))
        self.assertEqual(l.mock_requests,[]);self.assertEqual(l.spent_input,0)
        self.assertEqual(l.rejections[0]['reason'],'input_reservation_unavailable')
        self.assertNotIn('private fixture detail',json.dumps(l.rejections))

    @unittest.skipUnless(os.environ.get('EE_V3_CONTEXT_ASSETS'),'official pinned V4.1 offline tokenizer assets not supplied')
    def test_public_counter_counts_tools_results_and_reasoning_with_fixed_margin(self):
        c=PublicPromptCounter(Path(os.environ['EE_V3_CONTEXT_ASSETS']));base=request();initial=c.count(base)
        tool=dict(type='function',function=dict(name='toy',description='Short fixture.',parameters=dict(type='object',properties={})))
        with_tool=dict(base,tools=[tool]);declared=c.count(with_tool)
        a=dict(role='assistant',content='',reasoning_content='',tool_calls=[dict(id='call_1',type='function',function=dict(name='toy',arguments='{}'))])
        result=dict(role='tool',tool_call_id='call_1',content='x')
        history=dict(with_tool,messages=[*base['messages'],a,result]);n=c.count(history)
        changed=dict(with_tool,messages=[*base['messages'],dict(a,reasoning_content=' toy thought '*200),dict(result,content=' line \u96ea '*400)])
        estimate=c.reservation(changed)
        self.assertLess(initial,declared);self.assertLess(declared,n);self.assertLess(n,estimate['rendered_tokens'])
        self.assertEqual(estimate['margin_tokens'],max(MARGIN_MIN,(estimate['rendered_tokens']+9)//10))
        self.assertEqual(estimate['local_candidate_bound'],estimate['rendered_tokens']+estimate['margin_tokens'])
