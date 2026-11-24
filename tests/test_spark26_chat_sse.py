import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    import chat_sse_observer as wire


def choice(delta, finish=None, identity='chat-1'):
    return dict(id=identity, model='spark', object='chat.completion.chunk',
                choices=[dict(index=0,delta=delta,finish_reason=finish)])


def usage(completion=3, total=13):
    return dict(id='chat-1', model='spark', object='chat.completion.chunk', choices=[],
                usage=dict(prompt_tokens=10,completion_tokens=completion,total_tokens=total))


def pack(*events):
    return b''.join(b'data: '+json.dumps(row, ensure_ascii=False).encode()+b'\n\n' for row in events)


def read(data, **kwargs):
    clock=iter(x/1000 for x in range(1,1000))
    return wire.observe(io.BytesIO(data), model='spark', start=0.0, now=lambda:next(clock), **kwargs)


DONE=b'data: [DONE]\n\n'
STOP=choice({},'stop')


class ChatSSETests(unittest.TestCase):
    def test_role_is_not_payload_and_reasoning_is_not_answer(self):
        rows=pack(choice({'role':'assistant','content':''}), choice({'reasoning_content':'思考'}),
                  choice({'content':'答案'}), STOP, usage())
        result=read(rows+DONE)
        self.assertEqual((result.reasoning,result.text,result.tools),('思考','答案',[]))
        r=result.public
        self.assertEqual([r[x] for x in ('first_payload_ms','first_answer_ms','last_payload_ms','done_ms')],[4,6,6,12])
        self.assertEqual(r['payload_frames'],2)
        self.assertNotIn('答案',json.dumps(r,ensure_ascii=False))
        self.assertNotIn('ttft_ms',r)
        self.assertNotIn('tpot_ms',r)

    def test_wire_chunk_partitions_do_not_change_semantic_digest(self):
        fragments=pack(choice({'content':'蓝'}),choice({'content':'色'}),STOP,usage())+DONE
        inline=pack(choice({'content':'蓝色'}, 'stop'),usage())+DONE
        self.assertEqual(read(fragments).public['response_sha256'],read(inline).public['response_sha256'])
        self.assertEqual(read(fragments).public['payload_frames'],2)
        self.assertEqual(read(inline).public['payload_frames'],1)

    def test_tool_only_reassembles_json_and_omits_volatile_id(self):
        def stream(ident):
            initial=dict(index=0,type='function',id=ident,function=dict(name='record_color'))
            arg=lambda value: dict(index=0,function=dict(arguments=value))
            return pack(choice({'tool_calls':[initial]}),choice({'tool_calls':[arg('{"color":')]}),
                        choice({'tool_calls':[arg('"blue"}')]}),choice({},'tool_calls'),usage())+DONE
        a, b=read(stream('r1')),read(stream('r2'))
        self.assertEqual(a.tools,[dict(name='record_color',arguments={'color':'blue'})])
        self.assertEqual(a.public['response_sha256'],b.public['response_sha256'])
        self.assertIsNotNone(a.public['first_answer_ms'])
        self.assertNotIn('r1',json.dumps(a.public))

    def test_rejects_incomplete_or_cross_request_streams(self):
        content=choice({'content':'hi'})
        bad=(pack(content,STOP)+DONE, pack(content,STOP,usage()), pack(content,STOP,usage())+DONE[:-1],
             pack(content,STOP,usage(total=12))+DONE, pack(content, STOP,usage(completion=True,total=11))+DONE,
             pack(content,choice({'content':'x'},identity='chat-2'),STOP,usage())+DONE,
             pack(content,STOP,usage(),usage())+DONE,pack(STOP,usage())+DONE,
             pack(content,STOP,choice({'content':'late'}),usage())+DONE,
             pack(content,usage(),STOP)+DONE,pack(content,choice({},'error'),usage())+DONE,
             b'event: error\ndata: {"error": "test"}\n\n'+DONE)
        for data in bad:
            with self.subTest(wire_bytes=len(data)), self.assertRaises(ValueError):
                read(data)

    def test_refuses_incomplete_tool_args_and_hidden_usage(self):
        tool=dict(index=0,id='v',function=dict(name='test',arguments='{"a":1,"a":2}'))
        for bad in (choice({},'length'), choice({},'tool_calls')):
            with self.assertRaises(ValueError):
                read(pack(choice({'tool_calls':[tool]}),bad,usage())+DONE)
        row=choice({'content':'test'}); row['usage']=usage()['usage']
        with self.assertRaisesRegex(ValueError,'continuous usage'):
            read(pack(row,STOP,usage())+DONE)

    def test_multiline_sse_crlf_and_client_deadline(self):
        event=pack(choice({'content':'ok'},'stop'))
        event=event.replace(b'{"id": "chat-1", ',b'{"id": "chat-1",\ndata: ')
        data=(b': keepalive\n\n'+event+pack(usage())+DONE).replace(b'\n',b'\r\n')
        self.assertEqual(read(data).text,'ok')
        with self.assertRaisesRegex(ValueError,'wall-time'):
            read(data,timeout=0.002)
