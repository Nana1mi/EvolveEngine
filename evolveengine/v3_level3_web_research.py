"""Host-owned DeepSeek Search/Fetch and an append-only evidence ledger."""
from datetime import datetime, timezone
from decimal import Decimal
from ipaddress import ip_address
import json
import os
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError
from urllib.parse import urlsplit

from .v3_harness import canonical, digest


SCOPE = 'deepseek_open_v1'
USD_PER_SUCCESS = '0.00075'
PAGE = 8192
PAGE_BYTES = 32 << 20
KINDS = {'original', 'technology_transfer', 'upstream_patch'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_url(url):
    require(type(url) is str and 0 < len(url) <= 8192 and url == url.strip()
            and not any(ord(c) < 32 or c == '\\' for c in url), 'invalid public URL')
    parsed = urlsplit(url)
    host = parsed.hostname
    require(parsed.scheme in {'http', 'https'} and host and not parsed.username and not parsed.password,
            'fetch requires a public HTTP(S) URL')
    try:
        address = ip_address(host)
    except ValueError:
        hostname = host.rstrip('.').lower()
        require('.' in hostname and not hostname.endswith(('.localhost', '.local', '.internal', '.test', '.onion')),
                'fetch requires a public DNS name')
    else:
        require(address.is_global, 'fetch requires a public network address')
    return url


class WebResearch:
    def __init__(self, provider, root, cache=None):
        self.provider, self.root, self.cache = provider, Path(root), cache
        self.root.mkdir(exist_ok=False)
        self.calls, self.billable = 0, 0

    def _write(self, file, value):
        data = canonical(value)
        with file.open('xb') as out:
            out.write(data + b'\n')
            out.flush()
            os.fsync(out.fileno())
        return digest(data)

    def _call(self, operation, payload):
        self.calls += 1
        number = self.calls
        started = monotonic()
        self._write(self.root / f'{number:06}-request.json',
                    dict(operation=operation, payload=payload, utc=datetime.now(timezone.utc).isoformat()))
        status = None
        charge = None
        response_pin = None
        error = None
        try:
            with self.provider.request(f'/v1/tools/{operation}', payload=canonical(payload), timeout=60) as response:
                status = response.status
                raw = response.read(PAGE_BYTES + 1)
            require(status == 200 and len(raw) <= PAGE_BYTES, 'research response exceeds parser bounds')
            data = json.loads(raw)
            field = 'content' if operation == 'fetch' else 'results'
            expected = str if operation == 'fetch' else list
            require(type(data) is dict and type(data.get(field)) is expected, 'invalid research response shape')
            charge = USD_PER_SUCCESS if data[field] else '0'
            self.billable += bool(data[field])
            response_pin = self._write(self.root / f'{number:06}-response.json', data)
            return number, data
        except (HTTPError, OSError, ValueError) as exc:
            error = type(exc).__name__
            if isinstance(exc, HTTPError):
                status = exc.code
                charge = '0'
            raise ValueError(f'DeepSeek {operation} failed' + (f' with HTTP {status}' if status else '')) from exc
        finally:
            self._write(self.root / f'{number:06}-settlement.json',
                        dict(operation=operation, status=status, estimated_usd=charge, response_sha256=response_pin,
                             failure=error, elapsed_ms=round((monotonic()-started)*1000),
                             ended_utc=datetime.now(timezone.utc).isoformat()))

    def search(self, query, count=None, *, limit=None, offset=0, cached=False):
        if count is not None and limit is not None:
            require(type(count) is int and type(limit) is int and count == limit,
                    'conflicting count and limit in research_search')
        effective = count if count is not None else (limit if limit is not None else 10)
        if cached:
            require(self.cache is not None, 'no optional reference cache')
            require(type(effective) is int and effective >= 1, 'invalid cached search count')
            return self.cache.search(query, limit=min(effective, 12), offset=offset)
        require(type(query) is str and 0 < len(query) <= 128 and query.strip(), 'invalid DeepSeek query')
        require(type(effective) is int and 1 <= effective <= 50 and offset == 0, 'search returns up to 50 live results')
        number, data = self._call('search', dict(query=query, count=effective, need_content=False))
        results = [{k: row.get(k) for k in ('url','title','snippet','date','site_name')} for row in data['results']]
        return dict(trust='untrusted_reference', source=SCOPE, call=number, results=results)

    def fetch(self, url):
        url = public_url(url)
        number, data = self._call('fetch', dict(url=url))
        content = data['content']
        require(bool(content), 'no page content')
        record = dict(url=url, title=data.get('title'), content=content)
        identity = digest(canonical(record))
        path = self.root / f'page-{identity}.json'
        if not path.exists():
            self._write(path, record)
        return dict(self.read(identity), call=number)

    def read(self, identity, offset=0):
        require(type(identity) is str and len(identity) == 64 and
                all(c in '0123456789abcdef' for c in identity), 'invalid research object')
        path = self.root / f'page-{identity}.json'
        if not path.is_file():
            require(self.cache is not None, 'unknown research object')
            return self.cache.read(identity, offset)
        record = json.loads(path.read_bytes())
        require(digest(canonical(record)) == identity, 'research file changed')
        body = record.pop('content')
        require(type(offset) is int and 0 <= offset <= len(body), 'invalid research page offset')
        end = min(offset + PAGE, len(body))
        return dict(identity=identity, **record, text=body[offset:end], offset=offset,
                    next_offset=end if end < len(body) else None, trust='untrusted_reference', source=SCOPE)

    def record_reuse(self, kind, paths, url=None, revision=None, license=None, identity=None):
        require(kind in KINDS and type(paths) is list and all(type(p) is str for p in paths), 'invalid reuse record')
        if url is not None:
            public_url(url)
            require(all(type(s) is str and s.strip() for s in (revision, license)), 'record revision and license or mark unknown')
        else:
            require(kind == 'original' and identity is None, 'external reuse requires its source URL')
        if identity is not None:
            page = self.read(identity)
            require(page.get('url') == url, 'reuse record does not match archived source')
        row = dict(url=url, revision=revision, license=license, kind=kind, paths=paths, identity=identity)
        pin = digest(canonical(row))
        path = self.root / f'reuse-{pin}.json'
        if not path.exists():
            self._write(path, row)
        return dict(row, receipt=pin)

    def snapshot(self):
        return dict(calls=self.calls, billable_successes=self.billable, usd_per_success=USD_PER_SUCCESS,
                    estimated_usd=str(self.billable * Decimal(USD_PER_SUCCESS)),
                    billing='estimated_from_published_tariff')

    def page_refs(self):
        return tuple(path.stem.removeprefix('page-') for path in sorted(self.root.glob('page-*.json')))

    def import_pages(self, folder, identities):
        for identity in identities:
            require(type(identity) is str and len(identity) == 64 and
                    all(c in '0123456789abcdef' for c in identity), 'invalid prior research identity')
            source = folder / f'page-{identity}.json'
            require(source.is_file() and not source.is_symlink() and source.stat().st_size <= PAGE_BYTES+8192,
                    'prior research object is unavailable')
            page = json.loads(source.read_bytes())
            require(digest(canonical(page)) == identity, 'prior research object changed')
            dest = self.root / source.name
            if not dest.exists():
                self._write(dest,page)
