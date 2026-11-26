"""Stage the exact E0 CMake archives and pinned build-only Python wheels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


ARCHIVES = {
    'cutlass':('NVIDIA/cutlass','57e3cfb47a2d9e0d46eb6335c3dc411498efa198',
               '09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5'),
    'fmt':('fmtlib/fmt','553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28',
           'c314292789d28c3c3b420e75a7b2d1706f685f7fb63289128d46aeaea2c6be71'),
    'flashinfer':('flashinfer-ai/flashinfer','bc29697ba20b7e6bdb728ded98f04788e16ee021',
                  '931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d'),
}
WHEELS = {
    'scikit-build-core':('0.11.6','ce6d8fe64e6b4c759ea0fb95d2f8a68f60d2df31c2989838633b8ec930736360'),
    'setuptools-rust':('1.12.0','7e7db90547f224a835b45f5ad90c983340828a345554a9a660bdb2de8605dcdd'),
    'setuptools-scm':('9.2.2','30e8f84d2ab1ba7cb0e653429b179395d0c33775d54807fc5f1dd6671801aef7'),
    'packaging':('25.0','29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484'),
    'pathspec':('0.12.1','a0d503e138a4c123b27490a4f7beda6a01c6f288df0e4a8b79c7eb0dc7b4cc08'),
    'semantic-version':('2.10.0','de78a3b8e0feda74cabc54aab2da702113e33ac9d9eb9d2389bcf1f58b7d9177'),
    'wheel':('0.45.1','708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248'),
}
MIB = 1<<20
PUBLIC = {'pypi.org','files.pythonhosted.org','codeload.github.com'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('dependency URL redirected')


def fetch(url, path, cap, expected=None):
    p = urlsplit(url)
    need(p.scheme=='https' and p.netloc in PUBLIC and not p.query and not p.fragment,'unregistered dependency endpoint')
    request = build_opener(ProxyHandler({}),NoRedirect())
    h,n = sha256(),0
    with request.open(url,timeout=30) as reply, path.open('xb') as out:
        while chunk:=reply.read(1<<20):
            n+=len(chunk)
            need(n<=cap,'dependency download exceeds cap')
            h.update(chunk)
            out.write(chunk)
    need(n>0 and (expected is None or h.hexdigest()==expected),'dependency hash mismatch')
    return dict(url=url,sha256=h.hexdigest(),bytes=n)


def expand(archive, out, single_dir):
    with tarfile.open(archive,'r:gz') as source:
        members = source.getmembers()
        need(0 < len(members) <= 120000 and sum(m.size for m in members) < 1024*MIB,'dependency unpack bound exceeded')
        need(all(PurePosixPath(m.name).parts[:1]==(single_dir,) for m in members),'archive root mismatch')
        source.extractall(out,filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_root',type=Path)
    parser.add_argument('--archive-cache',type=Path,help='trusted operator staging; every byte is still hash checked')
    args = parser.parse_args()
    root = args.new_root
    root.mkdir(mode=0o700)
    for name in ('sources','wheels','downloads'):
        (root/name).mkdir()
    def archive(name):
        repo,commit,pin = ARCHIVES[name]
        path = root/'downloads'/f'{name}.tar.gz'
        url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
        if args.archive_cache is None:
            receipt = fetch(url,path,96*MIB,pin)
        else:
            cached = args.archive_cache/path.name
            need(cached.is_file() and not cached.is_symlink() and 0 < cached.stat().st_size <= 96*MIB,
                 'invalid pre-fetched archive')
            with cached.open('rb') as src, path.open('xb') as dst:
                shutil.copyfileobj(src,dst,length=MIB)
            with path.open('rb') as src:
                need(file_digest(src,'sha256').hexdigest()==pin,'pre-fetched archive hash mismatch')
            receipt = dict(url=url,sha256=pin,bytes=path.stat().st_size)
        single = f'{name}-{commit}'
        expand(path,root/'sources',single)
        (root/'sources'/single).rename(root/'sources'/name)
        return name,dict(**receipt,commit=commit)
    def wheel(name):
        version,pin = WHEELS[name]
        metadata = root/'downloads'/f'{name}.json'
        fetch(f'https://pypi.org/pypi/{name}/{version}/json',metadata,4*MIB)
        matching = [r for r in json.loads(metadata.read_bytes())['urls'] if r['digests']['sha256']==pin and not r['yanked']]
        need(len(matching)==1 and re.fullmatch(r'[A-Za-z0-9_.-]+-none-any\.whl',matching[0]['filename']),
             'registered universal wheel not found')
        row, = matching
        return name,dict(filename=row['filename'],version=version,
                         **fetch(row['url'],root/'wheels'/row['filename'],8*MIB,pin))
    with ThreadPoolExecutor(8) as pool:
        af = [pool.submit(archive,n) for n in ARCHIVES]
        wf = [pool.submit(wheel,n) for n in WHEELS]
        value = dict(sources=dict(f.result() for f in af),wheels=dict(f.result() for f in wf))
    with (root/'input-manifest.json').open('x',encoding='utf-8') as out:
        json.dump(value,out,sort_keys=True,separators=(',',':'))
        out.write('\n')


if __name__=='__main__':
    main()
