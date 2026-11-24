import contextlib
import io
import json
import unittest
from urllib.error import HTTPError

from evolveengine.v3_provider import GENERATION_TRUNCATED, LeaseError, Limits, MaxLease
from evolveengine.v3_runner import Gateway

MODEL='deepseek-flash'
STATS=dict(prompt_tokens=15,completion_tokens=9,total_tokens=24,completion_tokens_details={'reasoning_tokens':3})
REASON='分段\nline'


def event(*,delta=None,finish=None,usage=None,model=MODEL):
    j=dict(model=model,system_fingerprint='locked',choices=[] if delta is None and finish is None else [dict(index=0,delta=delta or {},finish_reason=finish)])
    if usage is not None:j['usage']=usage
    return b'data: '+json.dumps(j,ensure_ascii=False).encode()+b'\n\n'


def sse(*,tool=False,end='stop',usage=STATS,model=MODEL):
    parts=[event(delta=dict(role='assistant',reasoning_content='分段'),model=model),event(delta=dict(reasoning_content='\nline'),model=model)]
    if tool:
        end='tool_calls';parts.append(event(delta=dict(tool_calls=[dict(index=0,id='cid',type='function',function=dict(name='read',arguments='{}'))]),model=model))
    parts.extend((event(finish=end,model=model),event(usage=usage,model=model),b'data: [DONE]\n\n'))
    return b''.join(parts)


def req(extra=None,**kwargs):
    user=dict(role='user',content='toy task')
    return json.dumps(dict(model=MODEL,stream=True,thinking=dict(type='enabled'),reasoning_effort='max',max_tokens=20,
        messages=[user,*list(extra or [])])|kwargs,ensure_ascii=False).encode()


def checker(responses,**kwargs):
    call=[];res=iter(responses)
    @contextlib.contextmanager
    def transport(payload, remaining):
        call.append(json.loads(payload));body=next(res); yield body.splitlines(keepends=True)
    return MaxLease(Limits(2000,24,5,9000,16),transport,fingerprint='locked',**kwargs),call


class MaxProtocol(unittest.TestCase):
    def test_forward_max_and_exact_replay_without_double_count(self):
        first,second=sse(tool=True),sse();l,calls=checker([first,second]);self.assertEqual(l.exchange(req()),first)
        a=dict(role='assistant',content='',reasoning_content=REASON,tool_calls=[dict(id='cid',type='function',function=dict(name='read',arguments='{}'))]); tool=dict(role='tool',tool_call_id='cid',content='7')
        for invalid in (None, 'partial'):
            with self.assertRaises(LeaseError):l.exchange(req([dict(a,reasoning_content=invalid),tool]))
        self.assertEqual(l.exchange(req([a,tool])),second)
        self.assertEqual(calls[1]['messages'][1]['reasoning_content'],REASON)
        self.assertEqual((calls[0]['max_tokens'],calls[1]['max_tokens']),(16,15))
        self.assertTrue(all(c['stream_options']==dict(include_usage=True) and c['thinking']==dict(type='enabled') and c['reasoning_effort']=='max' for c in calls))
        self.assertEqual((l.spent_input,l.spent_output,sum(r['reasoning_tokens'] for r in l.rows)),(30,18,6))
        self.assertNotIn(REASON,json.dumps(l.rows,ensure_ascii=False))

    def test_wrong_model_and_non_text_refused_before_transport(self):
        l,calls=checker([])
        for request in (req(model='other'),req(reasoning_effort='high'),req(thinking=dict(type='disabled')),req(n=2),
                        req([dict(role='user',content=[dict(type='image',url='not-sent')])])):
            with self.assertRaises(LeaseError):l.exchange(request)
        self.assertEqual(calls,[])

    def test_closed_rejection_labels_separate_budget_and_replay_without_raw_text(self):
        l,calls=checker([sse(tool=True)])
        l.exchange(req());sentinel='SENSITIVE-FIXTURE-TEXT'
        a=dict(role='assistant',content='',reasoning_content=REASON,tool_calls=[dict(id='cid')])
        with self.assertRaises(LeaseError):l.exchange(req([dict(a,reasoning_content=sentinel)]))
        with self.assertRaises(LeaseError):l.exchange(req([a,dict(role='tool',tool_call_id='cid',content=sentinel*200)]))
        self.assertEqual([r['reason'] for r in l.rejections],['reasoning_replay','token_reservation'])
        self.assertTrue(all(not x['dispatched'] for x in l.rejections));self.assertEqual(len(calls),1)
        self.assertNotIn(sentinel,json.dumps(l.rejections));self.assertGreater(l.rejections[-1]['input_bound'],l.rejections[-1]['input_left'])

    def test_gateway_metadata_for_success_replay_and_budget(self):
        l,calls=checker([sse(tool=True),sse()]);g=Gateway(l,None,token='fixture')
        post=lambda raw:g.dispatch('/v1/chat/completions',raw,'Bearer fixture')
        self.assertEqual(post(req())[0],200)
        a=dict(role='assistant',content='',reasoning_content=REASON,tool_calls=[dict(id='cid',function=dict(name='PRIVATE-TOOL'))])
        tool=dict(role='tool',tool_call_id='cid',content='PRIVATE-CONTENT\u4e2d')
        bad=req([dict(a,reasoning_content='PRIVATE-REASON'),tool])
        self.assertEqual(post(bad),(400,'application/json',b'{"error":"model request unavailable"}'))
        too_much=req([a,dict(tool,content=tool['content']*200)])
        self.assertEqual(post(too_much)[0],400)
        good=req([a,tool]);self.assertEqual(post(good)[0],200)
        replay,budget=l.rejections
        self.assertEqual((replay['phase'],replay['profile'],replay['request_bytes']),('replay','pinned_max',len(bad)))
        self.assertEqual(replay['replay'],dict(expected=1,observed=1,missing=0,extra=0,reasoning_mismatches=1))
        self.assertIsNone(replay['input_bound']);self.assertIsNone(replay['output_bound'])
        self.assertEqual((budget['phase'],budget['input_left'],budget['output_left']),('reservation',1985,15))
        meta=l.rows[-1]['request_meta']
        self.assertEqual((meta['request_bytes'],meta['input_left'],meta['output_bound']),(len(good),1985,15))
        self.assertEqual(meta['replay']['reasoning_mismatches'],0)
        self.assertEqual(meta['recent_tool_results'],[dict(kind='other',text_utf8_bytes=len(tool['content'].encode()))])
        self.assertEqual(len(calls),2);self.assertNotIn('PRIVATE-',json.dumps([l.rows,l.rejections]))

    def test_gateway_pre_dispatch_failure_labels_have_no_usage_or_payload(self):
        l,calls=checker([]);g=Gateway(l,None,token='fixture')
        for raw in (b'P'*(l.limits.request_bytes+1),b'{PRIVATE',req(reasoning_effort='PRIVATE'),
                    req([dict(role=['PRIVATE'])]),req([dict(role='tool',content='SURROGATE')]).replace(b'SURROGATE',br'\ud800')):
            self.assertEqual(g.dispatch('/v1/chat/completions',raw,'Bearer fixture')[0],400)
        self.assertEqual([(r['reason'],r['phase']) for r in l.rejections],
                         [('request_bytes','request_size'),('malformed_request','parse'),('pinned_profile','profile'),
                          ('request_schema','replay'),('request_schema','replay')])
        self.assertIsNone(l.rejections[0]['profile']);self.assertEqual(l.rejections[2]['profile'],'mismatch')
        self.assertEqual(calls,[]);self.assertEqual((l.spent_input,l.spent_output),(0,0))
        self.assertNotIn('PRIVATE',json.dumps(l.rejections))

    def test_upstream_http_transport_verification_and_unknown_use_pre_request_headroom(self):
        cases=[(HTTPError('https://example.invalid/PRIVATE',429,'PRIVATE',None,io.BytesIO(b'PRIVATE')),
                'upstream_http','upstream_open',429),
               (OSError('PRIVATE'),'upstream_transport','upstream_open',None),
               (b'data: PRIVATE\n','upstream_verification','upstream_read',None),
               (RuntimeError('PRIVATE'),'unknown','upstream_open',None)]
        for fault,label,phase,status in cases:
            with self.subTest(label=label):
                def transport(*_):
                    if isinstance(fault,Exception):raise fault
                    return contextlib.nullcontext([fault])
                l=MaxLease(Limits(2000,24,5,9000,16),transport,fingerprint='locked');g=Gateway(l,None,token='fixture')
                self.assertEqual(g.dispatch('/v1/chat/completions',req(),'Bearer fixture')[0],400)
                item,=l.rejections
                self.assertEqual((item['reason'],item['phase'],item['http_status']),(label,phase,status))
                self.assertTrue(item['dispatched']);self.assertEqual(item['metered_turns'],0)
                self.assertEqual((item['input_left'],item['output_left']),(2000,24))
                self.assertEqual((l.spent_input,l.spent_output),(item['input_bound'],16))
                self.assertEqual(l.rows[0]['usage'],'uncertain_charged_reservation');self.assertTrue(l.poisoned)
                self.assertNotIn('PRIVATE',json.dumps([l.rows,l.rejections]))

    def test_incomplete_sse_consumes_reservation_and_prevents_retry(self):
        l,calls=checker([sse().replace(b'data: [DONE]\n\n',b'')]);
        for _ in range(2):
            with self.assertRaises(LeaseError):l.exchange(req())
        self.assertEqual(len(calls),1)
        self.assertEqual(l.rows[0]['usage'],'uncertain_charged_reservation')
        self.assertEqual(l.spent_output,16)

    def test_version_drift_records_returned_usage_once(self):
        l,calls=checker([sse(model='unexpected')]);
        with self.assertRaises(LeaseError):l.exchange(req())
        self.assertEqual(l.spent_output,9);self.assertEqual(l.rows[0]['usage'],'settled');self.assertTrue(l.poisoned)

    def test_length_and_reasoning_usage_validation(self):
        l,c=checker([sse(end='length')]);l.exchange(req());self.assertEqual(l.rows[0]['result'],GENERATION_TRUNCATED);self.assertTrue(l.poisoned)
        m,_=checker([sse(usage=dict(STATS,completion_tokens_details=dict(reasoning_tokens=11)))]);
        with self.assertRaises(LeaseError):m.exchange(req())
        self.assertEqual(m.spent_output,16)

    def test_r02_length_is_settled_single_turn_cap_not_aggregate_or_denial(self):
        usage=dict(prompt_tokens=13318,completion_tokens=4096,total_tokens=17414,
                   completion_tokens_details=dict(reasoning_tokens=4096))
        body=sse(end='length',usage=usage);calls=[]
        def transport(payload,_):
            calls.append(json.loads(payload))
            return contextlib.nullcontext(body.splitlines(keepends=True))
        l=MaxLease(Limits(200000,24000,28,9000,4096),transport,fingerprint='locked')
        g=Gateway(l,None,token='fixture')
        raw=req([dict(role='user',content='x'*16000)],max_tokens=8192)
        # The response to the original request is HTTP 200; length is a billed stop.
        status,kind,reply=g.dispatch('/v1/chat/completions',raw,'Bearer fixture')
        self.assertEqual((status,kind),(200,'text/event-stream'))
        chunks=[line.removeprefix(b'data: ') for line in reply.splitlines() if line.startswith(b'data: ')]
        self.assertEqual(chunks[-1],b'[DONE]')
        terminal=json.loads(chunks[0]);self.assertEqual(len(chunks),2)
        self.assertEqual(terminal['choices'],[dict(index=0,delta={},finish_reason='length')]);self.assertEqual(terminal['usage'],usage)
        self.assertEqual(calls[0]['max_tokens'],4096)
        self.assertEqual((l.spent_input,l.spent_output),(13318,4096))
        self.assertEqual((l.rows[0]['result'],l.rows[0]['usage'],l.rows[0]['reasoning_tokens']),(GENERATION_TRUNCATED,'settled',4096))
        self.assertTrue(l.poisoned);self.assertEqual(l.rejections,[]);self.assertEqual(g.events,[])

    def test_output_reservation_clips_and_denies_when_empty(self):
        payloads=[]
        @contextlib.contextmanager
        def transport(payload,_):
            data=json.loads(payload);payloads.append(data)
            n=data['max_tokens'];usage=dict(prompt_tokens=15,completion_tokens=n,total_tokens=15+n)
            yield sse(usage=usage).splitlines(keepends=True)
        l=MaxLease(Limits(200000,24000,28,9000,8192),transport,fingerprint='locked')
        for _ in range(3):l.exchange(req(max_tokens=8192))
        self.assertEqual([p['max_tokens'] for p in payloads],[8192,8192,7616]);self.assertEqual(l.spent_output,24000)
        without_max=json.loads(req());del without_max['max_tokens']
        for request in (req(max_tokens=8192),json.dumps(without_max).encode()):
            with self.assertRaises(LeaseError):l.exchange(request)
        self.assertEqual(len(payloads),3);self.assertEqual(l.spent_output,24000)
        self.assertEqual([(r['reason'],r['output_left'],r['output_bound']) for r in l.rejections],[('token_reservation',0,0)]*2)

    def test_length_never_exposes_tool_delta_or_records_its_id(self):
        partial=dict(id='cid',index=0,type='function',function=dict(name='read',arguments='{"path":'))
        body=event(delta=dict(reasoning_content='PRIVATE-THINKING',tool_calls=[partial]))+event(finish='length',usage=STATS)+b'data: [DONE]\n\n'
        l,calls=checker([body]);reply=l.exchange(req())
        self.assertNotIn(b'PRIVATE-THINKING',reply);self.assertNotIn(b'tool_calls',reply);self.assertNotIn(b'cid',reply)
        self.assertEqual(l.seen_ids,{});self.assertEqual(l.spent_output,9);self.assertEqual(l.rows[0]['result'],GENERATION_TRUNCATED)
        with self.assertRaises(LeaseError):l.exchange(req())
        self.assertEqual(len(calls),1);self.assertEqual(l.spent_output,9)

    def test_overrun_is_uncertain_and_not_added_twice(self):
        class SmallInput(MaxLease):
            def _input_ceiling(self,req,tools,meta):return 10
        for n,m in ((11,9),(8,17)):
            calls=[]
            def transport(payload,_):
                calls.append(payload)
                return contextlib.nullcontext(sse(usage=dict(prompt_tokens=n,completion_tokens=m,total_tokens=n+m)).splitlines(keepends=True))
            l=SmallInput(Limits(100,100,4,9000,16),transport,fingerprint='locked')
            for _ in range(2):
                with self.assertRaises(LeaseError):l.exchange(req())
            self.assertEqual(len(calls),1)
            row,=l.rows
            self.assertEqual((row['usage'],row['result']),('usage_uncertain','usage_uncertain'))
            self.assertEqual((row['reported_input_tokens'],row['reported_output_tokens']),(n,m))
            self.assertEqual((row['input_overrun'],row['output_overrun']),(max(0,n-10),max(0,m-16)))
            self.assertEqual((l.spent_input,l.spent_output),(max(n,10),max(m,16)))
            self.assertEqual(l.rejections[0]['reason'],'usage_uncertain');self.assertTrue(l.poisoned)

    def test_fragment_claim_derived_from_actual_wire(self):
        l,c=checker([sse()],expected_prompt_fragments=('h-mark',))
        with self.assertRaises(LeaseError):l.exchange(req())
        l.exchange(req([dict(role='system',content='h-mark\nonly example')]))
        self.assertEqual(len(l.rows[0]['prompt_fragments']),1)

if __name__=='__main__':unittest.main()
