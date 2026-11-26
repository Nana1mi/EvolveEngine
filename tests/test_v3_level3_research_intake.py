from email.message import Message
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evolveengine.v3_harness import digest
from evolveengine.v3_level3_research import PublicCorpus
from experiments.spark26_gguf.level3 import stage_research as s


class FakeResponse(BytesIO):
    status=200

    def __init__(self, raw, mime, final_url):
        super().__init__(raw)
        self.headers=Message()
        self.headers['Content-Type']=mime
        self.final_url=final_url

    def geturl(self):
        return self.final_url


class ResearchIntakeTests(TestCase):
    def setUp(self):
        temp=TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base=Path(temp.name).resolve()
        self.spool=self.base/'spool'
        self.raw={
            'guide':b'<head><meta charset="utf-8"></head><body><nav>Navigation</nav><p>CUDA guide.</p>'+b'Guide '*2000+b'</body>',
            'paper':b'<article><p>arXiv:2407.08608v1 License: CC BY 4.0</p><h1>Introduction</h1>'+b'Attention '*1200+b'<h2>Conclusion</h2></article>',
            'source':b'/* SPDX-License-Identifier: BSD-3-Clause */\n'+b'// MMA source\n'*900,
            'license':b'Redistribution and use in source and binary forms',
        }

    def fake(self, request, *, timeout):
        url=request.full_url
        key=next(k for k,v in s.FETCH.items() if v[0]==url)
        self.assertEqual(timeout,30)
        return FakeResponse(self.raw[key],s.FETCH[key][1],url)

    def review(self):
        body={n:digest((self.spool/f'{n}.body').read_bytes()) for n in s.REFS}
        path=self.base/'review.json'
        return path,s._json(path,dict(scope=s.REVIEW_SCOPE,reviewer='test',references=body))

    def test_pins_replayed_and_exact_commit(self):
        s.fetch(self.spool,self.fake)
        review,pin=self.review()
        sealed=self.base/'sealed'
        result=s.seal(self.spool,sealed,review,pin)
        corpus=PublicCorpus(sealed,result)
        refs=corpus.search('CUTLASS MMA')['results']
        entry=next(r for r in refs if r['kind']=='source')
        self.assertEqual(entry['revision'],s.CUTLASS)
        self.assertEqual(entry['sha256'],digest(self.raw['source']))
        self.assertTrue(corpus.read(entry['sha256'])['truncated'])
        with self.assertRaises(ValueError):
            s.seal(self.spool,self.base/'unapproved',review,'a'*64)
        (self.spool/'source.body').write_bytes(self.raw['source']+b'changed')
        with self.assertRaisesRegex(ValueError,'changed after'):
            s.seal(self.spool,self.base/'drift',review,pin)

    def test_redirect_and_unreviewed_target_body_fail_closed(self):
        def redirect(request,*,timeout):
            return FakeResponse(self.raw['guide'],'text/html','https://example.org/redirect')
        with self.assertRaisesRegex(ValueError,'scope or type mismatch'):
            s.fetch(self.base/'redirected',redirect)
        row=json.loads((self.base/'redirected/guide.receipt.json').read_bytes())
        self.assertEqual(row['status'],'incomplete')
        self.raw['guide']+=b'<p>SGLang version after cutoff</p>'
        with self.assertRaisesRegex(ValueError,'unreviewed SGLang'):
            s.fetch(self.spool,self.fake)
        self.assertFalse((self.spool/'paper.raw').exists())
