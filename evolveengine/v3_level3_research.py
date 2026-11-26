"""Search/page a pinned, host-reviewed corpus without fetching an Agent URL.

Intake is a trusted pre-run operation; reviewers must check derived SGLang
material. Lexical rejection is an extra constraint, not semantic review.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from .v3_level3_source import ARCHIVE_ROOT, SGLANG_ARCHIVE_SHA256, SGLANG_REVISION, read_source_archive

SHA = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
SGLANG = re.compile(r'\bsgl[\s_-]*ang\b|\bsgl[_-]kernel\b', re.I)
PAGE = 8192
MIB = 1 << 20


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _id(value, pattern=SHA):
    return type(value) is str and pattern.fullmatch(value) is not None


def _read(root, relative, cap):
    path = root/relative
    need(path.is_file() and not path.is_symlink() and path.resolve() == path,
         'research object missing or redirected')
    with path.open('rb') as f:
        raw = f.read(cap+1)
    need(len(raw) <= cap, 'research object exceeds bound')
    return raw


def _canonical_url(url):
    need(type(url) is str and 0 < len(url) <= 1500 and '\\' not in url
         and all(32 < ord(ch) < 127 for ch in url), 'invalid public reference URL')
    parts = urlsplit(url)
    host = parts.hostname or ''
    # Admit DNS names only. This code never resolves names or follows redirects.
    need(parts.scheme == 'https' and parts.netloc == host and not (parts.query or parts.fragment)
         and re.fullmatch(r'([a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}', host) is not None
         and not host.endswith(('.local','.internal','.localhost','.test')), 'not a canonical public HTTPS URL')
    return parts


@dataclass(frozen=True)
class AncestorFence:
    """Prove ancestor status from a trusted local object graph, never commit dates."""
    repository: Path
    e0: str = SGLANG_REVISION

    def __call__(self, commit):
        if not _id(commit,COMMIT) or not _id(self.e0,COMMIT):
            return False
        cmd = [*self._prefix(),'merge-base','--is-ancestor',commit,self.e0]
        try:
            return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=15).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _prefix(self):
        return ['git','--no-replace-objects','-C',str(self.repository.resolve())]

    def verify_blob(self, commit, path, expected_size, expected_sha):
        if not self(commit) or not re.fullmatch(r'[a-zA-Z0-9_./+-]+',path):
            return False
        spec = commit+':'+path
        command = dict(stdin=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
        try:
            if int(subprocess.check_output([*self._prefix(),'cat-file','-s',spec], **command)) != expected_size:
                return False
            raw = subprocess.check_output([*self._prefix(),'cat-file','blob',spec], **command)
            return len(raw) == expected_size and sha256(raw).hexdigest() == expected_sha
        except (OSError, ValueError, subprocess.SubprocessError):
            return False


@dataclass(frozen=True)
class Reference:
    title: str
    url: str
    kind: str
    revision: str | None
    scope: str
    sha256: str
    bytes: int
    fetched_utc: str
    license: str
    review_sha256: str

    def public(self):
        return dict(title=self.title, url=self.url, kind=self.kind, revision=self.revision, sha256=self.sha256,
                    bytes=self.bytes, fetched_utc=self.fetched_utc, license=self.license)


class PinnedE0Archive:
    """Exact upstream E0 blobs from the previously registered archive; no Git clone required."""
    e0 = SGLANG_REVISION

    def __init__(self, path):
        self.files = read_source_archive(path, expected_sha256=SGLANG_ARCHIVE_SHA256,
                                         expected_root=ARCHIVE_ROOT).files

    def verify_blob(self, commit, path, expected_size, expected_sha):
        raw = self.files.get(path)
        return (commit == self.e0 and raw is not None and len(raw) == expected_size and
                sha256(raw).hexdigest() == expected_sha)


class PublicCorpus:
    def __init__(self, root, manifest_sha, *, ancestor=None, e0_archive=None):
        self.root = Path(root).resolve(strict=True)
        self.identity, self.ancestor, self._entries = manifest_sha, ancestor, {}
        raw = _read(self.root, 'manifest.json',128*1024)
        need(_id(manifest_sha) and sha256(raw).hexdigest() == manifest_sha, 'research manifest drift')
        manifest = json.loads(raw)
        need(type(manifest) is dict and set(manifest) == {'schema','references'}
             and manifest['schema'] == 'level3-public-corpus-v1' and type(manifest['references']) is list
             and 0 < len(manifest['references']) <= 64, 'invalid research index')
        total = 0
        for item in manifest['references']:
            need(type(item) is dict and set(item) == set(Reference.__dataclass_fields__), 'invalid reference entry')
            ref = Reference(**item)
            if ref.scope == 'sglang_pinned' and self.ancestor is None and e0_archive is not None:
                self.ancestor = PinnedE0Archive(e0_archive)
            self._admit(ref)
            total += ref.bytes
            need(total <= 24*MIB and ref.sha256 not in self._entries, 'research corpus size or duplicate')
            self._body(ref)
            self._entries[ref.sha256] = ref

    def _admit(self, ref):
        parts = _canonical_url(ref.url)
        need(_id(ref.sha256) and _id(ref.review_sha256) and type(ref.bytes) is int and 0 < ref.bytes <= MIB
             and all(type(x) is str and 0 < len(x) <= 300 and '\n' not in x and '\r' not in x
                     for x in (ref.title,ref.fetched_utc,ref.license))
             and ref.kind in ('web','paper','source'), 'invalid provenance metadata')
        if ref.kind == 'source':
            match = re.fullmatch(r'/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/(?:blob/|commit/)([0-9a-f]{40})(?:/[a-zA-Z0-9_./+-]+)?', parts.path)
            need(parts.hostname == 'github.com' and match is not None and all(x not in ('.','..') for x in parts.path.split('/'))
                 and _id(ref.revision,COMMIT) and match[3] == ref.revision, 'source must name an exact commit')
        else:
            need(ref.revision is None or (type(ref.revision) is str and 0 < len(ref.revision) <= 100), 'invalid version')
        exact = ref.kind == 'source' and parts.path.startswith('/sgl-project/sglang/')
        if exact or ref.scope == 'sglang_pinned':
            file = re.fullmatch(r'/sgl-project/sglang/blob/[0-9a-f]{40}/([a-zA-Z0-9_./+-]+)',parts.path)
            need(exact and file is not None and ref.scope == 'sglang_pinned' and self.ancestor is not None
                 and self.ancestor.e0 == SGLANG_REVISION
                 and self.ancestor.verify_blob(ref.revision,file[1],ref.bytes,ref.sha256),
                 'SGLang blob not proven at or before E0')
        else:
            need(ref.scope == 'general_reviewed' and not SGLANG.search(ref.title+' '+ref.url),
                 'SGLang reference needs exact upstream provenance')

    def _body(self, ref):
        raw = _read(self.root,f'sha256/{ref.sha256}',MIB)
        need(len(raw) == ref.bytes and sha256(raw).hexdigest() == ref.sha256, 'reference content drift')
        text = raw.decode('utf-8')
        need('\0' not in text and (ref.scope == 'sglang_pinned' or not SGLANG.search(text)),
             'unversioned SGLang reference or unsupported payload')
        return raw

    def search(self, query, limit=6, offset=0):
        need(type(query) is str and 0 < len(query) <= 320 and type(limit) is int and 1 <= limit <= 64 and
             type(offset) is int and offset >= 0, 'invalid search request')
        terms = set(re.findall(r'[^\W_]+',query.casefold()))
        need(0 < len(terms) <= 16, 'invalid search terms')
        hits = []
        for ref in self._entries.values():
            title, text = ref.title.casefold(), self._body(ref).decode('utf-8').casefold()
            score = sum(3*(term in title)+(term in text) for term in terms)
            if score:
                hits.append((-score,ref.sha256,ref))
        page = sorted(hits)[offset:offset+limit]
        end = offset + len(page)
        return dict(scope='reviewed_public_cache',corpus=self.identity,trust='untrusted_reference',
                    total=len(hits), offset=offset, next_offset=end if end < len(hits) else None,
                    results=[ref.public() for _,_,ref in page])

    def read(self, identity, offset=0):
        need(_id(identity) and identity in self._entries, 'unknown or non-admitted public reference')
        ref = self._entries[identity]
        raw = self._body(ref)
        need(type(offset) is int and 0 <= offset <= len(raw) and (offset == len(raw) or raw[offset]&0xc0 != 0x80),
             'offset must be a UTF-8 start byte')
        end = min(len(raw),offset+PAGE)
        while end < len(raw) and raw[end]&0xc0 == 0x80:
            end -= 1
        return dict(scope='reviewed_public_cache',corpus=self.identity,trust='untrusted_reference',source=ref.public(),
                    text=raw[offset:end].decode('utf-8'),offset=offset,truncated=end < len(raw),
                    next_offset=end if end < len(raw) else None)
