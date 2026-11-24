"""Offline-only P3 context candidate. A public renderer is not a hosted billing oracle."""
from pathlib import Path
import json
from types import MappingProxyType

from .v3_provider import LeaseError, MaxLease, canonical, digest, need

VERSION='p3-context-1017-paged-read-v1-candidate'
MODEL_REVISION='08c8657f02291e11b6f6b63e0e16014f8e748518'
ARTIFACTS=MappingProxyType({
    'encoding.py':'13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2',
    'tokenizer.json':'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b',
})
TOKENIZERS_VERSION='0.22.1'
# Predeclared engineering margin for offline comparison, not a proof of a bound on hosted usage.
MARGIN_MIN=512
MARGIN_PERCENT=10


class PublicPromptCounter:
    def __init__(self, assets):
        from tokenizers import Tokenizer, __version__
        need(__version__==TOKENIZERS_VERSION,'counter dependency does not match protocol')
        blobs={name:(Path(assets)/name).read_bytes() for name in ARTIFACTS}
        need(all(digest(data)==ARTIFACTS[name] for name,data in blobs.items()),'counter artifact does not match protocol')
        scope={'__name__':'_pinned_deepseek_v41_encoding'}
        exec(compile(blobs['encoding.py'],'deepseek-v4.1-1017/encoding.py','exec'),scope)
        self.render=scope['encode_messages'];self.tokenizer=Tokenizer.from_str(blobs['tokenizer.json'].decode('utf-8'))
        self.tokenizer.no_truncation();self.tokenizer.no_padding()

    def count(self,request):
        """Exact count for the declared public text rendering, not exact API usage."""
        # Match sorting/layout of the actual JSON delivered by the trusted broker.
        wire=json.loads(canonical(request));messages=wire['messages'];tools=wire.get('tools',[])
        if tools:
            if messages[0]['role']!='system':messages.insert(0,dict(role='system',content=''))
            messages[0]['tools']=tools
        need(not wire.get('response_format') and all(m['role'] in {'system','user','assistant','tool'} for m in messages),
             'unsupported public counter request')
        text=self.render(messages,thinking_mode='thinking',drop_thinking=False,reasoning_effort=wire['reasoning_effort'])
        return len(self.tokenizer.encode(text,add_special_tokens=False).ids)

    def reservation(self,request):
        n=self.count(request)
        margin=max(MARGIN_MIN,(n*MARGIN_PERCENT+99)//100)
        return dict(rendered_tokens=n,margin_tokens=margin,local_candidate_bound=n+margin)


class OfflineContextLease(MaxLease):
    """Do not wire this candidate to a hosted provider; it has no hosted-bound evidence."""
    def __init__(self,limits,counter,*,responses,**kwargs):
        self.counter=counter;self.assistant_hashes=[]
        # Only predetermined local SSE bytes; do not accept a paid/outbound transport.
        self.responses=iter(tuple(responses));self.mock_requests=[]
        from contextlib import nullcontext
        def local_transport(wire,_timeout):
            raw=next(self.responses)
            need(type(raw) is bytes,'invalid offline fixture')
            self.mock_requests.append(digest(wire))
            return nullcontext(raw.splitlines(keepends=True))
        super().__init__(limits,local_transport,**kwargs)

    def _input_ceiling(self,req,tools,meta):
        try:parts=self.counter.reservation(req)
        except (AssertionError,ValueError,KeyError,TypeError,IndexError):
            raise LeaseError('input reservation unavailable') from None
        meta['input_reservation']=parts
        return parts['local_candidate_bound']

    def _record_reasoning(self,ids,reason):
        super()._record_reasoning(ids,reason)
        self.assistant_hashes.append(reason)

    def _replay_matches(self,messages,prior):
        assistants=[m for m in messages if m['role']=='assistant']
        if not all(type(m.get('reasoning_content')) is str for m in assistants):return False
        return (super()._replay_matches(messages,prior) and
                [digest(m['reasoning_content'].encode()) for m in assistants]==self.assistant_hashes)
