"""Run the loopback-only, read-only EvolveEngine RSI Observatory."""
import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sqlite3
import threading
from time import monotonic
from urllib.parse import urlsplit

from . import observatory_data as projection

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments/spark26_gguf/level3/evidence/exploration-results-20261125.json'
FRONTEND = ROOT / 'observatory/dist'
SSE_TIMEOUT = 55  # Allow periodic graceful reconnect even behind an idle proxy.
POLL = 2.0
HEADERS = dict(**{'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
                  'Cross-Origin-Resource-Policy': 'same-origin', 'X-Frame-Options': 'DENY',
                  'Content-Security-Policy': "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                             "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
                                             "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"})


def encode(doc):
    return json.dumps(doc, separators=(',', ':'), ensure_ascii=True, allow_nan=False, sort_keys=True).encode('ascii')


class Store:
    def __init__(self, live, *, archive=ARCHIVE, published=None):
        self.live = live
        self.published = published or {}
        self.archive = projection.archive_runs(archive) if archive is not None and archive.is_file() else {}
        if (self.live.keys() & self.archive.keys() or self.published.keys() & (self.live.keys() | self.archive.keys())):
            raise ValueError('local run id conflicts with an archive')
        self.cache = {}
        self.locks = {name: threading.Lock() for name in self.live.keys() | self.published.keys()}

    def contains(self, name):
        return name in self.live or name in self.archive or name in self.published

    def runs(self):
        # Listing never reads a private run. Status is in its own sanitized snapshot.
        return [dict(id=k, label=k.replace('-', ' ').upper(), mode='controller') for k in (*self.live, *self.published)] + [
            {k: row[k] for k in ('id', 'label', 'mode')} for row in self.archive.values()]

    def published_snapshot(self, name):
        # Only a trusted producer may populate these explicitly configured files.
        # There is deliberately no browser upload or ability to select a file.
        path = self.published[name]
        if not path.is_file():
            return projection.empty(name, name.replace('-', ' ').upper(), 'controller')
        if path.resolve() != path:
            raise ValueError('aliased publication path')
        doc = projection.read_json(path, cap=32 << 20)
        if doc.get('id') != name or doc.get('schema_version') != 1 or doc.get('mode') not in {'controller', 'archive'}:
            raise ValueError('invalid published snapshot identity')
        return doc

    @staticmethod
    def payload(doc):
        raw = encode(doc)
        revision = sha256(raw).hexdigest()
        return revision, encode(dict(doc, revision=revision))

    def snapshot(self, name):
        if name in self.archive:
            return self.payload(self.archive[name])
        with self.locks[name]:
            if name in self.cache and self.cache[name][0] + POLL > monotonic():
                return self.cache[name][1]
            try:
                out = (self.published_snapshot(name) if name in self.published
                       else projection.local_run(self.live[name], name))
            except (OSError, ValueError, sqlite3.Error, TypeError, RecursionError):
                # A transient read never destroys a last good snapshot or writes the source.
                if name in self.cache:
                    return self.cache[name][1]
                out = projection.empty(name, name.replace('-', ' ').upper(), 'controller')
                out['notices'].append('Run metadata is temporarily unreadable.')
            result = self.payload(out)
            self.cache[name] = (monotonic(), result)
            return result


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, store, static=FRONTEND, viewer_origin=None):
        if addr[0] != '127.0.0.1':
            raise ValueError('The dashboard must be behind a loopback listener')
        if viewer_origin is not None:
            url = urlsplit(viewer_origin)
            if (url.scheme != 'https' or not url.hostname or url.username is not None or url.password is not None
                    or url.path or url.query or url.fragment or url.netloc != url.netloc.lower()):
                raise ValueError('viewer origin must be one exact HTTPS origin without a slash')
        super().__init__(addr, Handler)
        self.viewer_origin = viewer_origin
        self.store = store
        self.static = static.resolve()
        self.stream_slots = threading.BoundedSemaphore(8)
        self.closing = threading.Event()

    def server_close(self):
        self.closing.set()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(12)

    def handle(self):
        try:
            super().handle()
        except ConnectionError:
            # A browser can close a keep-alive socket after receiving the response.
            self.close_connection = True

    def log_message(self, *_):
        pass  # No query strings, browser headers, or run identifiers in logs.

    def response_headers(self, code, kind, *, size=None, cache='no-store'):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', cache)
        if self.headers.get('Content-Length', '0') != '0' or self.headers.get('Transfer-Encoding'):
            self.send_header('Connection', 'close')
            self.close_connection = True
        for key, value in HEADERS.items():
            self.send_header(key, value)
        if size is not None:
            self.send_header('Content-Length', str(size))

    def respond(self, body, kind='application/json; charset=utf-8', code=200, cache='no-store'):
        self.response_headers(code, kind, size=len(body), cache=cache)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def fail(self, code):
        self.respond(encode({'error': {403: 'local same-origin access only', 404: 'not found',
                     405: 'read only', 503: 'stream unavailable'}.get(code, 'request failed')}), code=code)

    def route(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        host, origin = self.headers_in('Host'), self.headers_in('Origin')
        viewer = self.server.viewer_origin
        origins = {f'http://{host}'}
        if viewer and host == urlsplit(viewer).netloc:
            hosts.add(host)
            origins = {viewer}
        if host not in hosts or (origin is not None and origin not in origins):
            return self.fail(403)
        if self.headers_in('Sec-Fetch-Site') in {'cross-site', 'same-site'}:
            return self.fail(403)
        if (self.command not in {'GET', 'HEAD'} or self.headers_in('Transfer-Encoding')
                or self.headers.get('Content-Length', '0') != '0'):
            return self.fail(405)
        url = urlsplit(self.path)
        if url.scheme or url.netloc or (url.query and url.path != '/'):
            return self.fail(404)
        path = url.path
        if path == '/api/runs':
            return self.respond(encode(dict(schema_version=1, runs=self.server.store.runs())))
        if match := re.fullmatch(r'/api/runs/([a-z0-9][a-z0-9_-]{0,47})/(snapshot|events)', path):
            name, endpoint = match.groups()
            if not self.server.store.contains(name):
                return self.fail(404)
            if endpoint == 'snapshot':
                return self.respond(self.server.store.snapshot(name)[1])
            if self.command != 'GET':
                return self.fail(405)
            return self.stream(name)
        if path == '/':
            name = 'index.html'
        elif re.fullmatch(r'/assets/[A-Za-z0-9_.-]+\.(?:js|css|woff2|svg)', path):
            name = path[1:]
        else:
            return self.fail(404)
        file = self.server.static / name
        if file.resolve() != file or not file.is_file():
            return self.fail(404)
        kind = {'js': 'text/javascript; charset=utf-8', 'css': 'text/css; charset=utf-8',
                'html': 'text/html; charset=utf-8'}.get(file.suffix.lstrip('.')) or mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
        return self.respond(file.read_bytes(), kind=kind, cache='no-cache' if path == '/' else 'public,max-age=31536000,immutable')

    def headers_in(self, name):
        return self.headers.get(name)

    def stream(self, name):
        if not self.server.stream_slots.acquire(blocking=False):
            return self.fail(503)
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            for key, value in HEADERS.items():
                self.send_header(key, value)
            self.end_headers()
            self.close_connection = True
            seen = self.headers_in('Last-Event-ID')
            start = monotonic()
            self.wfile.write(b'retry: 2000\n\n')
            while not self.server.closing.is_set() and monotonic() - start < SSE_TIMEOUT:
                revision, body = self.server.store.snapshot(name)
                if revision != seen:
                    self.wfile.write(b'id: '+revision.encode('ascii')+b'\nevent: snapshot\ndata: '+body+b'\n\n')
                    seen = revision
                else:
                    self.wfile.write(b': unchanged\n\n')
                self.wfile.flush()
                self.server.closing.wait(POLL)
        finally:
            self.server.stream_slots.release()

    def do_GET(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
            self.close_connection = True

    do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_GET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8848)
    p.add_argument('--run', action='append', default=[], metavar='ID=PATH',
                   help='register a single trusted Controller run root, optionally before it exists')
    p.add_argument('--published', action='append', default=[], metavar='ID=JSON',
                   help='follow a sanitized snapshot atomically published by a trusted producer')
    p.add_argument('--no-archive', action='store_true')
    p.add_argument('--viewer-origin', help='exact HTTPS origin served by an authenticated organization reverse proxy; does not enable LAN binding')
    a = p.parse_args()
    live, published = {}, {}
    for raw, registry in [(r, live) for r in a.run] + [(r, published) for r in a.published]:
        name, sep, path = raw.partition('=')
        if not sep or not path or not projection.RUN_ID.fullmatch(name) or name in live or name in published:
            p.error('expected a unique lowercase ID=PATH')
        registry[name] = Path(path).absolute() if registry is published else Path(path).resolve()
    if not (FRONTEND / 'index.html').is_file():
        p.error('build the UI first with npm --prefix observatory ci && npm --prefix observatory run build')
    store = Store(live, archive=None if a.no_archive else ARCHIVE, published=published)
    with Server(('127.0.0.1', a.port), store, viewer_origin=a.viewer_origin) as server:
        print(f'RSI Observatory: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
