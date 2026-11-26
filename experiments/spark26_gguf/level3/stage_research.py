"""Intake three fixed public sources into a private, reviewed Level 3 corpus.

Host-only tool: no caller-controlled URL, redirects, credentials, model use,
or handing off raw HTTP. `fetch` stages witnesses; a separate pinned review
is required for `seal`. Both operate outside the Agent and GPU directories.
"""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import build_opener, HTTPRedirectHandler, Request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_research import MIB, SGLANG, PinnedE0Archive, PublicCorpus, need
from evolveengine.v3_level3_source import SGLANG_ARCHIVE_SHA256, SGLANG_REVISION

CUTLASS = '57e3cfb47a2d9e0d46eb6335c3dc411498efa198'
CUTLASS_PATH = 'include/cutlass/arch/mma_sm80.h'
GUIDE = 'https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html'
PAPER = 'https://arxiv.org/html/2407.08608v1'
FETCH = {
    'guide': (GUIDE, 'text/html', 2*MIB),
    'paper': (PAPER, 'text/html', 2*MIB),
    'source': (f'https://raw.githubusercontent.com/NVIDIA/cutlass/{CUTLASS}/{CUTLASS_PATH}', 'text/plain', MIB),
    'license': (f'https://raw.githubusercontent.com/NVIDIA/cutlass/{CUTLASS}/LICENSE.txt', 'text/plain', 32768),
}
REFS = {
    'guide': ('CUDA C++ Best Practices Guide 12.8',GUIDE,'web','12.8.0','NVIDIA documentation; no redistribution asserted'),
    'paper': ('FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision',PAPER,'paper',
              '2407.08608v1','CC-BY-4.0; Shah et al.; visible-text projection of arXiv HTML'),
    'source': ('NVIDIA CUTLASS architecture MMA source',f'https://github.com/NVIDIA/cutlass/blob/{CUTLASS}/{CUTLASS_PATH}',
               'source',CUTLASS,'BSD-3-Clause (retained source header and LICENSE.txt)'),
}
REVIEW_SCOPE = 'generic_public_reference_no_sglang_history'
NVIDIA_PERFORMANCE = 'https://docs.nvidia.com/deeplearning/performance/'
GPU_GUIDES = {
    'gpu-performance': ('NVIDIA GPU Performance Background',
                        NVIDIA_PERFORMANCE+'dl-performance-gpu-background/index.html'),
    'gemm-performance': ('NVIDIA Matrix Multiplication Background',
                         NVIDIA_PERFORMANCE+'dl-performance-matrix-multiplication/index.html'),
}

# Host-reviewed upstream paths, exact bytes only. These are the base tree, not
# candidate patches, later commits, benchmark results or the E0 model overlay.
Q = 'python/sglang/kernels/aot/csrc/quantization/gguf/'
RUNNER = 'python/sglang/srt/model_executor/runner/'
E0_REFERENCES = {
    'python/sglang/srt/layers/quantization/gguf.py': 'Q8_0 decode and prefill dispatch',
    Q+'gguf_kernel.cu': 'GGUF CUDA kernel dispatch',
    Q+'mmvq.cuh': 'Quantized matrix-vector multiply for single-request decode',
    Q+'mmq.cuh': 'Quantized matrix-matrix multiply for prefill',
    Q+'dequantize.cuh': 'Quantized block dequantization',
    'python/sglang/kernels/aot/include/sgl_kernel/quantization/gguf/vecdotq.cuh': 'Quantized vector dot products',
    'python/sglang/kernels/ops/attention/decode_attention.py': 'Decode attention kernels',
    RUNNER+'decode_cuda_graph_runner.py': 'CUDA Graph decode execution',
    RUNNER+'prefill_cuda_graph_runner.py': 'CUDA Graph prefill execution',
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*_args,**_kwargs):
        return None


class VisibleText(HTMLParser):
    IGNORE = frozenset(('script','style','head','nav','footer','noscript','svg'))
    BLOCK = frozenset(('section','h1','h2','h3','h4','p','tr','div','pre','li','figcaption','br'))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.chunks = []

    def handle_starttag(self,tag,attrs):
        if tag in self.IGNORE:
            self.hidden.append(tag)
        elif not self.hidden and tag in self.BLOCK:
            self.chunks.append('\n')

    def handle_endtag(self,tag):
        if self.hidden:
            if self.hidden[-1]==tag:
                self.hidden.pop()
        elif tag in self.BLOCK:
            self.chunks.append('\n')

    def handle_data(self,data):
        if not self.hidden:
            self.chunks.append(data)


def content(raw,mime):
    if mime=='text/plain':
        raw.decode('utf-8')
        return raw
    parser = VisibleText()
    parser.feed(raw.decode('utf-8'))
    parser.close()
    return ('\n'.join(s.strip() for s in ''.join(parser.chunks).split('\n') if s.strip())+'\n').encode('utf-8')


def _write(path,raw):
    with path.open('xb') as out:
        out.write(raw)


def _json(path,row):
    raw = canonical(row)+b'\n'
    _write(path,raw)
    return digest(raw)


def fetch(root, opener=None, *, guides=False):
    root.mkdir(mode=0o700)
    request = opener if opener is not None else build_opener(NoRedirect()).open
    targets = {k:(url,'text/html',2*MIB) for k,(_,url) in GPU_GUIDES.items()} if guides else FETCH
    for name,(url,mime,cap) in targets.items():
        record = dict(retrieval_url=url,attempt=1,utc=datetime.now(timezone.utc).isoformat(),paid_api_calls=0)
        try:
            with request(Request(url,headers={'User-Agent':'EvolveEngine-Level3-reference-intake/1',
                                               'Accept-Encoding':'identity'}),timeout=30) as response:
                need(response.geturl()==url and response.status==200 and response.headers.get_content_type()==mime and
                     response.headers.get('Content-Encoding','identity')=='identity','public response scope or type mismatch')
                raw = response.read(cap+1)
                need(0 < len(raw) <= cap,'public response exceeds retrieval allowance')
                record.update(response_status=response.status,content_type=mime,etag=response.headers.get('ETag'),
                              last_modified=response.headers.get('Last-Modified'),bytes=len(raw),sha256=digest(raw))
            _write(root/f'{name}.raw',raw)
            if name in REFS or name in GPU_GUIDES:
                visible = content(raw,mime)
                need(8*1024 < len(visible) <= MIB and not SGLANG.search(visible.decode('utf-8')),
                     'unreviewed SGLang reference, truncated body, or reference too large')
                _write(root/f'{name}.body',visible)
                record.update(body_bytes=len(visible),body_sha256=digest(visible),
                              projection='visible_text_v1' if mime=='text/html' else 'verbatim_blob')
            record['status']='fetched'
        except Exception as exc:
            record.update(status='incomplete',error=type(exc).__name__)
            raise
        finally:
            _json(root/f'{name}.receipt.json',record)


def seal(spool,root,review,review_pin):
    raw = review.read_bytes()
    seal_record = json.loads(raw)
    need(digest(raw)==review_pin and type(seal_record) is dict and seal_record.keys()=={'scope','reviewer','references'} and
         seal_record['scope']==REVIEW_SCOPE and type(seal_record['reviewer']) is str and seal_record['reviewer'].strip() and
         type(seal_record['references']) is dict and seal_record['references'].keys()==REFS.keys(),
         'missing or mismatched trusted corpus review')
    references, verified = [], {}
    for name,(title,url,kind,revision,license_text) in REFS.items():
        receipt = json.loads((spool/f'{name}.receipt.json').read_bytes())
        source = (spool/f'{name}.raw').read_bytes()
        body = (spool/f'{name}.body').read_bytes()
        address = digest(body)
        need(receipt['status']=='fetched' and receipt['retrieval_url']==FETCH[name][0] and receipt['sha256']==digest(source) and
             len(source)==receipt['bytes'] and body==content(source,FETCH[name][1]) and
             address==receipt['body_sha256']==seal_record['references'][name] and len(body)==receipt['body_bytes'] and
             not SGLANG.search(body.decode('utf-8')),'reference changed after trusted review')
        if name=='paper':
            need(b'arXiv:2407.08608v1' in body and b'License: CC BY 4.0' in body and
                 b'Introduction' in body and b'Conclusion' in body,'paper body, edition, or attribution missing')
        elif name=='source':
            need(b'SPDX-License-Identifier: BSD-3-Clause' in body,'exact source blob lacks license notice')
        verified[address] = body
        references.append(dict(title=title,url=url,kind=kind,revision=revision,scope='general_reviewed',sha256=address,
                               bytes=len(body),fetched_utc=receipt['utc'],license=license_text,review_sha256=review_pin))
    license_raw = (spool/'license.raw').read_bytes()
    license_record = json.loads((spool/'license.receipt.json').read_bytes())
    need(license_record['status']=='fetched' and license_record['retrieval_url']==FETCH['license'][0] and
         digest(license_raw)==license_record['sha256'] and b'Redistribution and use' in license_raw,'source license missing')
    root.mkdir(mode=0o700)
    (root/'sha256').mkdir(mode=0o700)
    for address,body in verified.items():
        _write(root/'sha256'/address,body)
    pin = _json(root/'manifest.json',dict(schema='level3-public-corpus-v1',references=references))
    PublicCorpus(root,pin) # verify sealed candidate-independent bytes before publishing the pin
    return pin


def _reviewed_guides(spool, review, review_sha256):
    review_raw = review.read_bytes()
    data = json.loads(review_raw)
    need(digest(review_raw)==review_sha256 and data.get('scope')==REVIEW_SCOPE
         and isinstance(data.get('reviewer'),str) and data['reviewer'].strip()
         and set(data.get('references',{}))==GPU_GUIDES.keys(), 'expected a pinned private review of both guides')
    for name,(title,url) in GPU_GUIDES.items():
        receipt = json.loads((spool/f'{name}.receipt.json').read_bytes())
        body = (spool/f'{name}.body').read_bytes()
        raw = (spool/f'{name}.raw').read_bytes()
        address = digest(body)
        need(receipt['retrieval_url']==url and receipt['status']=='fetched' and receipt['sha256']==digest(raw)
             and address==data['references'][name]==receipt['body_sha256'] and body==content(raw,'text/html')
             and not SGLANG.search(body.decode('utf-8')), 'reviewed guide content drift')
        yield dict(title=title,url=url,kind='web',revision=None,scope='general_reviewed',sha256=address,
                   bytes=len(body),fetched_utc=receipt['utc'],review_sha256=review_sha256,
                   license='NVIDIA documentation; no redistribution asserted'), body


def extend_e0(base, base_pin, archive, output, *, guides=None):
    """Publish a new corpus atomically; the old reviewed corpus stays addressable."""
    source = PinnedE0Archive(archive)
    PublicCorpus(base, base_pin, ancestor=source)
    refs = json.loads((base/'manifest.json').read_bytes())['references']
    pins = {path:digest(source.files[path]) for path in E0_REFERENCES}
    review = dict(scope='upstream_e0_blobs', archive_sha256=SGLANG_ARCHIVE_SHA256,
                  revision=SGLANG_REVISION, references=pins)
    output = output.resolve()
    need(not output.exists(), 'new corpus output already exists')
    with TemporaryDirectory(dir=output.parent) as folder:
        stage = Path(folder)/'corpus'
        objects = stage/'sha256'
        objects.mkdir(parents=True)
        review_pin = _json(stage/'e0-index-review.json', review)
        for ref in refs:
            body = (base/'sha256'/ref['sha256']).read_bytes()
            need(digest(body)==ref['sha256'], 'base corpus changed during extension')
            _write(objects/ref['sha256'], body)
        if guides is not None:
            for ref,body in _reviewed_guides(*guides):
                _write(objects/ref['sha256'],body)
                refs.append(ref)
        for path,title in E0_REFERENCES.items():
            body, address = source.files[path], pins[path]
            _write(objects/address, body)
            refs.append(dict(title='SGLang E0 · '+title, kind='source', scope='sglang_pinned',
                             url=f'https://github.com/sgl-project/sglang/blob/{SGLANG_REVISION}/{path}',
                             revision=SGLANG_REVISION, sha256=address, bytes=len(body), review_sha256=review_pin,
                             fetched_utc=datetime.now(timezone.utc).isoformat(),
                             license='Upstream source; retain file headers and applicable upstream notices.'))
        pin = _json(stage/'manifest.json',dict(schema='level3-public-corpus-v1', references=refs))
        PublicCorpus(stage, pin, ancestor=source)
        stage.rename(output)
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    modes.add_parser('fetch').add_argument('spool',type=Path)
    modes.add_parser('fetch-guides').add_argument('spool',type=Path)
    s = modes.add_parser('seal')
    for arg in ('spool','output','review'):
        s.add_argument(arg,type=Path)
    s.add_argument('--review-sha256',required=True)
    x = modes.add_parser('extend-e0', help='Add the registered upstream decode and prefill sources')
    for arg in ('base','archive','output'):
        x.add_argument(arg,type=Path)
    x.add_argument('--base-sha256',required=True)
    x.add_argument('--guide-spool',type=Path)
    x.add_argument('--guide-review',type=Path)
    x.add_argument('--guide-review-sha256')
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode=='fetch':
        fetch(args.spool)
    elif args.mode=='fetch-guides':
        fetch(args.spool,guides=True)
    elif args.mode=='extend-e0':
        guides = (args.guide_spool,args.guide_review,args.guide_review_sha256)
        need(not any(guides) or all(guides), 'supply all three guide-review inputs')
        print(extend_e0(args.base,args.base_sha256,args.archive,args.output,guides=guides if all(guides) else None))
    else:
        print(seal(args.spool,args.output,args.review,args.review_sha256))


if __name__=='__main__':
    main()
