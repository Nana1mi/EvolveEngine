"""DEV-only helper to equalize initial prefix-cache state across cancellation arms.

Not part of live v2. It preserves overlap within each arm: the two peers share
one fresh namespace, while the complete and cancel arms never share a namespace.
Only the first usable response counter is evidence of initial state. Later cache
activity belongs to the tested scheduler and is not forced to match.
"""
from contextlib import contextmanager
import json
import re
from uuid import uuid4


class CacheParityUnavailable(ValueError):
    pass


class _Response:
    def __init__(self, response, owner):
        self._response, self._owner = response, owner
        self.status, self.headers = response.status, response.headers

    def readline(self, limit):
        line = self._response.readline(limit)
        if self._owner.first_counter is None and line.startswith(b'data:') and len(line) < limit:
            body = line[5:].strip()
            if body != b'[DONE]':
                try:
                    meta = json.loads(body)['meta_info']
                    cached, tokens, rid = meta['cached_tokens'], meta['completion_tokens'], meta['id']
                except (KeyError, TypeError, ValueError) as exc:
                    raise CacheParityUnavailable('missing initial cache witness') from exc
                if (type(cached) is not int or cached < 0 or type(tokens) is not int or tokens < 0
                        or rid != self._owner.rid or cached > self._owner.prompt_tokens):
                    raise CacheParityUnavailable('invalid initial cache witness')
                # Pre-output frames may precede scheduling the initial prefill.
                # Use the first frame that actually carries generated progress.
                if tokens > 0:
                    self._owner.first_counter = cached
        return line


class ScopedStream:
    """Adapter to the trusted stream observer's existing call(route, payload) interface."""
    def __init__(self, transport, salt):
        self.transport, self.salt = transport, salt
        self.rid = self.prompt_tokens = self.first_counter = None

    @contextmanager
    def call(self, route, payload):
        if route == 'abort_request':
            if self.rid is None or payload != {'rid': self.rid, 'abort_all': False}:
                raise CacheParityUnavailable('cache scope cannot abort another request')
            with self.transport(route,payload) as reply:
                yield reply
            return
        if (route != 'generate' or self.rid is not None or 'cache_salt' in payload or
                not isinstance(payload.get('rid'),str) or not re.fullmatch(r'[0-9a-f]{32}',payload['rid']) or
                type(payload.get('input_ids')) is not list or not payload['input_ids'] or payload.get('stream') is not True):
            raise CacheParityUnavailable('one bound streaming generate required')
        self.rid, self.prompt_tokens = payload['rid'], len(payload['input_ids'])
        with self.transport(route, dict(payload,cache_salt=self.salt)) as reply:
            yield _Response(reply,self)
        if self.first_counter is None:
            raise CacheParityUnavailable('no cache witness after generation began')


class ColdPairs:
    """One oracle allocation per arm; verifier never treats missing cache state as zero."""
    def __init__(self, transport):
        self.transport, self.scopes, self.salts = transport, {}, set()

    def arm(self, trigger, arm):
        key=(trigger,arm)
        if type(trigger) is not int or trigger <= 0 or arm not in ('complete','cancel') or key in self.scopes:
            raise CacheParityUnavailable('invalid or reused cold arm')
        salt=uuid4().hex
        if salt in self.salts:
            raise CacheParityUnavailable('reused cache namespace')
        self.salts.add(salt)
        self.scopes[key]=tuple(ScopedStream(self.transport,salt) for _ in range(2))
        return self.scopes[key]

    def witness(self, triggers):
        expected={(k,a) for k in triggers for a in ('complete','cancel')}
        if not expected or set(self.scopes)!=expected:
            raise CacheParityUnavailable('missing cold pair arm')
        result=[]; rids=set()
        for (trigger,arm), peers in sorted(self.scopes.items()):
            counters=[]
            for peer in peers:
                if peer.rid is None or peer.rid in rids or peer.first_counter != 0:
                    raise CacheParityUnavailable('cold parity unverified; no signed comparison')
                rids.add(peer.rid); counters.append(peer.first_counter)
            result.append(dict(trigger=trigger,arm=arm,cached_tokens_initial=counters))
        return result
