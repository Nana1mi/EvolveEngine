from contextlib import contextmanager
from io import BytesIO
import json
from unittest import TestCase
from unittest.mock import patch

from experiments.spark26_gguf.level3.isolation_cache_parity import CacheParityUnavailable, ColdPairs, ScopedStream


def payload(rid):
    return dict(rid=f'{rid:032x}',input_ids=[100,101],stream=True)


class Reply(BytesIO):
    status=200
    headers={}


class CacheParityTests(TestCase):
    def transport(self, *, cached=0, mode='normal'):
        self.sent=[]
        @contextmanager
        def call(route, data):
            self.sent.append((route,data))
            if route=='abort_request':
                yield Reply()
                return
            def frame(tokens, count):
                row=dict(id=data['rid'] if mode!='misbound' else '0'*32, completion_tokens=tokens)
                if mode!='absent': row['cached_tokens']=count
                return b'data: '+json.dumps(dict(meta_info=row)).encode()+b'\n'
            raw = b'data: [DONE]\n' if mode=='empty' else frame(0,1)+frame(1,cached)+frame(2,2)
            yield Reply(raw)
        return call

    def read(self, peer, n, *, cancel=False):
        request=payload(n)
        with peer.call('generate',request) as reply:
            self.assertEqual(reply.status,200)
            while reply.readline(4096): pass
            if cancel:
                with peer.call('abort_request',dict(rid=request['rid'],abort_all=False)): pass
        self.assertNotIn('cache_salt',request)

    def test_only_within_arm_the_salt_is_shared(self):
        pairs=ColdPairs(self.transport())
        for index,arm in enumerate(('cancel','complete')):
            for i,peer in enumerate(pairs.arm(8,arm)):
                self.read(peer,index*2+i+1,cancel=arm=='cancel' and i==1)
        calls=[row for route,row in self.sent if route=='generate']
        self.assertEqual(calls[0]['cache_salt'],calls[1]['cache_salt'])
        self.assertEqual(calls[2]['cache_salt'],calls[3]['cache_salt'])
        self.assertNotEqual(calls[1]['cache_salt'],calls[2]['cache_salt'])
        rows=pairs.witness((8,)); self.assertEqual([r['cached_tokens_initial'] for r in rows],[[0,0],[0,0]])
        with self.assertRaisesRegex(CacheParityUnavailable,'reused'): pairs.arm(8,'cancel')

    def test_invalid_or_missing_counters_fail_before_a_comparison(self):
        for mode,cache in (('empty',0),('absent',0),('misbound',0),('normal',True),('normal',3)):
            with self.subTest(mode=mode,cache=cache), self.assertRaises(CacheParityUnavailable):
                self.read(ScopedStream(self.transport(cached=cache,mode=mode),'test'),1)
        pairs=ColdPairs(self.transport(cached=1))
        for i,arm in enumerate(('cancel','complete')):
            for j,peer in enumerate(pairs.arm(8,arm)): self.read(peer,1+2*i+j)
        with self.assertRaisesRegex(CacheParityUnavailable,'parity unverified'): pairs.witness((8,))

    def test_scope_reuse_and_wrong_cancel_are_rejected(self):
        peer=ScopedStream(self.transport(),'test')
        self.read(peer,1)
        with self.assertRaisesRegex(CacheParityUnavailable,'one bound'): self.read(peer,2)
        with self.assertRaisesRegex(CacheParityUnavailable,'another request'):
            with peer.call('abort_request',dict(rid=payload(2)['rid'],abort_all=False)): pass
        with patch('experiments.spark26_gguf.level3.isolation_cache_parity.uuid4') as mock:
            mock.return_value.hex='a'*32; pairs=ColdPairs(self.transport()); pairs.arm(8,'complete')
            with self.assertRaisesRegex(CacheParityUnavailable,'namespace'): pairs.arm(8,'cancel')
