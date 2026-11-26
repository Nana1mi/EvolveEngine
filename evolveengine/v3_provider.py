"""V3's independent DeepSeek Flash Max lease; no legacy Thinking Off overrides.

The trusted caller owns the transport and credentials. `exchange()` accepts only the text
chat protocol and buffers one metered turn. Complete turns pass through unchanged;
a verified length stop is forwarded without incomplete model output.
"""
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException
import json
from threading import Lock
import time
from urllib.error import HTTPError

from .v3_harness import canonical

MODEL='deepseek-flash'
THINK={'type':'enabled'}
DIAGNOSTIC_VERSION='max-lease-metadata-v3'
GENERATION_TRUNCATED='generation_truncated'
TOOL_KINDS=frozenset({'read','edit','write','bash','ee_rpc','grep','find','ls'})
# Only closed labels leave this trusted component. Never record request text, tool
# arguments, reasoning content, or an upstream error body.
REJECTION_CODES={
    'provider lease unavailable':'turn_or_poisoned_lease',
    'provider wall lease exhausted':'wall_lease',
    'model request size':'request_bytes',
    'malformed model request':'malformed_request',
    'wrong pinned model/Max profile':'pinned_profile',
    'prior tool reasoning or identity mismatch':'reasoning_replay',
    'text or output token reservation exceeds remaining cap':'token_reservation',
    'registered Harness prompt fragment missing':'harness_prompt',
    'input reservation unavailable':'input_reservation_unavailable',
    'full generation window unavailable':'full_generation_window_unavailable',
    'context cannot fit full generation window':'context_generation_window',
    'API cost reservation exceeds remaining RMB':'api_cost_limit',
}



class LeaseError(ValueError):
    pass


def need(condition,message):
    if not condition:raise LeaseError(message)


def digest(b):return sha256(b).hexdigest()


@dataclass(frozen=True)
class Limits:
    input: int
    output: int
    turns: int
    wall_ms: int
    per_turn_output: int
    request_bytes: int=512*1024

    def __post_init__(self):
        need(all(type(v) is int and v>0 for v in (self.input,self.output,self.turns,self.wall_ms,self.per_turn_output,self.request_bytes)), 'positive integer caps required')


@dataclass(frozen=True)
class MeteredLimits(Limits):
    """Uncapped totals for a separately registered engineering exploration."""
    def __post_init__(self):
        need(all(v is None for v in (self.input, self.output, self.turns, self.wall_ms)),
             'metered totals must be explicitly uncapped')
        need(all(type(v) is int and v > 0 for v in (self.per_turn_output, self.request_bytes)),
             'positive per-request limits required')


def counts(usage):
    need(type(usage) is dict,'missing authoritative stream usage')
    a,b,t=(usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens'))
    need(all(type(x) is int and x>=0 for x in (a,b,t)) and t==a+b,'invalid token totals')
    detail=usage.get('completion_tokens_details') or {}
    need(type(detail) is dict,'invalid output detail')
    q=detail.get('reasoning_tokens');need(q is None or type(q) is int and 0<=q<=b,'invalid reasoning subset')
    return a,b,q


def allowed_messages(msgs):
    need(type(msgs) is list and msgs,'missing messages')
    ids={}
    for m in msgs:
        need(type(m) is dict and m.get('role') in {'system','developer','user','assistant','tool'},'unknown message role')
        content=m.get('content')
        need(content is None or type(content) is str or (type(content) is list and all(type(x) is dict and x.get('type')=='text' and type(x.get('text')) is str and set(x)<={'type','text'} for x in content)),'not a supported text-only request')
        if m['role']=='assistant' and m.get('tool_calls'):
            reason=m.get('reasoning_content');need(reason is None or type(reason) is str,'invalid tool-call reasoning content')
            h=digest((reason or '').encode())
            need(type(m['tool_calls']) is list,'invalid replay tool call list')
            for v in m['tool_calls']:
                need(type(v) is dict,'invalid replay tool call')
                cid=v.get('id');need(type(cid) is str and cid and cid not in ids,'invalid or duplicate replay tool ID');ids[cid]=h
    return ids


def tool_result_metadata(messages):
    """Project validated messages; keep only a bounded suffix and fixed tool labels."""
    by_id={};recent=[];count=0
    for msg in messages:
        if msg['role']=='assistant' and msg.get('tool_calls'):
            for call in msg['tool_calls']:
                function=call.get('function')
                name=function.get('name') if type(function) is dict else None
                by_id[call['id']]=name if type(name) is str and name in TOOL_KINDS else 'other'
        if msg['role']=='tool':
            tid=msg.get('tool_call_id');content=msg.get('content')
            kind=by_id.get(tid,'unmatched') if type(tid) is str else 'unmatched'
            size=(len(content.encode('utf-8')) if type(content) is str else
                  sum(len(part['text'].encode('utf-8')) for part in content or ()))
            count+=1;recent.append(dict(kind=kind,text_utf8_bytes=size))
            if len(recent)>8:recent.pop(0)
    return count,recent


class MaxLease:
    context_window = None
    require_full_generation_window = False
    record_generation_timing = False
    finish_reasons = frozenset({'stop', 'tool_calls', 'length', 'content_filter'})
    sse_bytes_per_token = 256

    def sse_byte_limit(self, token_cap):
        return token_cap * self.sse_bytes_per_token + 2**20

    def __init__(self,limits,transport,*,fingerprint=None, clock=time.monotonic, expected_prompt_fragments=()):
        self.limits=limits;self.transport=transport;self.fingerprint=fingerprint;self.clock=clock
        self.deadline=None if limits.wall_ms is None else clock()+limits.wall_ms/1000;self.spent_input=self.spent_output=0;self.rows=[];self.rejections=[];self.seen_ids={};self.poisoned=False
        self._lock=Lock();self._expected_prompt_fragments=tuple(expected_prompt_fragments)

    def _left(self):
        if self.deadline is None:
            return float('inf')
        left=self.deadline-self.clock();need(left>0,'provider wall lease exhausted');return left

    def _read(self,up,ceiling):
        data=bytearray(); usage=None; reasons=[]; ids={};models=set();fps=set();finish=None;done=False
        first_delta = last_delta = None
        for line in up:
            self._left();need(isinstance(line,bytes) and len(data)+len(line)<=self.sse_byte_limit(ceiling),'SSE size reservation exceeded')
            data.extend(line)
            if not line.startswith(b'data:'):continue
            piece=line[5:].strip()
            if not piece:continue
            need(not done,'duplicate terminal or late SSE data')
            if piece==b'[DONE]':done=True;continue
            c=json.loads(piece);need(type(c) is dict,'invalid stream chunk')
            if c.get('model'):models.add(c['model'])
            if c.get('system_fingerprint'):fps.add(c['system_fingerprint'])
            if c.get('usage') is not None:
                need(usage is None,'duplicate metered stream usage');usage=c['usage']
            choices=c.get('choices',[]);need(type(choices) is list and len(choices)<=1,'multiple assistant choices prohibited')
            for z in choices:
                need(type(z) is dict,'invalid stream choice')
                need(z.get('index',0)==0 and finish is None,'invalid choice sequence')
                d=z.get('delta',{}) or {};need(type(d) is dict,'invalid stream delta');v=d.get('reasoning_content')
                if v is not None:need(type(v) is str,'invalid reasoning delta');reasons.append(v)
                calls=d.get('tool_calls') or [];need(type(calls) is list,'invalid stream tools list')
                for x in calls:
                    need(type(x) is dict,'invalid stream tool call')
                    if x.get('id'):
                        ix=x.get('index');need(type(x['id']) is str and type(ix) is int and ix>=0 and (ix not in ids or ids[ix]==x['id']),'ambiguous tool identity');ids[ix]=x['id']
                if self.record_generation_timing and (v or d.get('content') or calls):
                    last_delta = self.clock()
                    if first_delta is None:
                        first_delta = last_delta
                if z.get('finish_reason') is not None:finish=z['finish_reason']
        self._left();need(done and finish in self.finish_reasons,'missing or unsupported SSE completion')
        need(len(set(ids.values()))==len(ids),'duplicate response tool ID')
        need(finish!='tool_calls' or ids,'incomplete tool call response')
        if first_delta is not None:
            self.rows[-1]['generation_timing'] = dict(first_delta_monotonic=first_delta, last_delta_monotonic=last_delta)
        return bytes(data),usage,models,fps,finish,digest(''.join(reasons).encode()),bool(reasons),list(ids.values())

    def _input_ceiling(self,req,tools,meta):
        return len(canonical([req['messages'],tools]))+256

    def _replay_matches(self,messages,prior):
        return prior==self.seen_ids

    def _record_reasoning(self,ids,reason):
        self.seen_ids.update({x:reason for x in ids})

    def exchange(self,raw):
        """Transport is called with canonical bytes and remaining seconds. Unknown usage burns its reservation."""
        with self._lock:
            old_turns=len(self.rows)
            meta=dict(request_bytes=len(raw) if type(raw) is bytes else None,
                      input_left=None if self.limits.input is None else max(0,self.limits.input-self.spent_input),
                      output_left=None if self.limits.output is None else max(0,self.limits.output-self.spent_output),
                      input_bound=None,output_bound=None,requested_max_tokens=None,upstream_request_bytes=None,profile=None,replay=None,
                      tool_result_count=None,recent_tool_results=None)
            stage={'phase':'lease'}
            try:return self._do(raw,meta,stage)
            except (ValueError,TypeError,RecursionError,RuntimeError) as exc:
                if len(self.rejections)<32:
                    reason=stage.get('upstream_reason')
                    if reason is None:
                        if isinstance(exc,LeaseError):reason=REJECTION_CODES.get(str(exc),'request_schema')
                        elif (stage['phase'] in {'replay','reservation','wire'} and
                              isinstance(exc,(ValueError,TypeError,RecursionError))):reason='request_schema'
                        else:reason='unknown'
                    self.rejections.append(dict(reason=reason,phase=stage['phase'],metered_turns=old_turns,
                       dispatched=len(self.rows)!=old_turns,http_status=stage.get('http_status'),**meta))
                if isinstance(exc,LeaseError):raise
                raise LeaseError('model request unavailable; see trusted diagnostics') from None

    def _authorize_upstream(self, meta):
        pass

    def _settle_upstream(self, row):
        pass

    def _do(self,raw,meta,stage):
        need(not self.poisoned and (self.limits.turns is None or len(self.rows)<self.limits.turns),'provider lease unavailable')
        self._left()
        stage['phase']='request_size';need(type(raw) is bytes and 0<len(raw)<=self.limits.request_bytes,'model request size')
        stage['phase']='parse'
        try:req=json.loads(raw)
        except (ValueError,RecursionError) as e:raise LeaseError('malformed model request') from e
        stage['phase']='profile'
        profile_ok=(type(req) is dict and req.get('model')==MODEL and req.get('stream') is True and
                    req.get('thinking')==THINK and req.get('reasoning_effort')=='max' and req.get('n',1)==1)
        meta['profile']='pinned_max' if profile_ok else 'mismatch'
        need(profile_ok,'wrong pinned model/Max profile')
        stage['phase']='replay'
        prior=allowed_messages(req.get('messages'))
        common=prior.keys() & self.seen_ids.keys()
        meta['replay']=dict(expected=len(self.seen_ids),observed=len(prior),missing=len(self.seen_ids.keys()-prior.keys()),
                            extra=len(prior.keys()-self.seen_ids.keys()),reasoning_mismatches=sum(prior[x]!=self.seen_ids[x] for x in common))
        meta['tool_result_count'],meta['recent_tool_results']=tool_result_metadata(req['messages'])
        need(self._replay_matches(req['messages'],prior),'prior tool reasoning or identity mismatch')
        stage['phase']='reservation'
        tools=req.get('tools',[]);need(type(tools) is list,'invalid tool inventory')
        bound=self._input_ceiling(req,tools,meta)
        need(type(bound) is int and bound>0,'input reservation unavailable')
        meta['input_bound']=bound
        cap=self.limits.per_turn_output
        if self.limits.output is not None:
            cap=min(cap,self.limits.output-self.spent_output)
        if self.context_window is not None:
            cap = min(cap, self.context_window - bound - getattr(self, 'context_margin', 0))
        if 'max_tokens' in req:
            asked=req['max_tokens'];need(type(asked) is int and asked>0,'invalid turn output request')
            meta['requested_max_tokens']=asked;cap=min(cap,asked)
        meta['output_bound']=cap
        need(self.limits.input is None or bound<=self.limits.input-self.spent_input,'text or output token reservation exceeds remaining cap')
        if getattr(self, 'minimum_generation', 0):
            need(cap >= self.minimum_generation, 'context cannot fit full generation window')
        if self.require_full_generation_window:
            full = self.limits.per_turn_output
            need(meta['requested_max_tokens'] == full and self.limits.output-self.spent_output >= full,
                 'full generation window unavailable')
            need(cap == full, 'context cannot fit full generation window')
        need(cap>0,'text or output token reservation exceeds remaining cap')
        stage['phase']='wire'
        req.pop('max_completion_tokens',None);req.update(max_tokens=cap,stream_options=dict(include_usage=True))
        wire=canonical(req)
        systems='\n'.join(m['content'] for m in req['messages'] if m['role']=='system' and isinstance(m.get('content'),str))
        loaded=[]
        for fragment in self._expected_prompt_fragments:
            need(fragment and fragment in systems,'registered Harness prompt fragment missing')
            loaded.append(digest(fragment.encode()))
        meta['upstream_request_bytes']=len(wire)
        timeout=self._left()
        stage['phase']='cost_reservation'
        self._authorize_upstream(meta)
        row=dict(turn=len(self.rows)+1, model=MODEL, thinking='enabled', effort='max', upstream_sha=digest(wire),
                 prior_tool_contexts=len(prior), toolset_sha=digest(canonical(tools)), prompt_fragments=loaded,
                 input_reserved=bound, output_reserved=cap, request_meta=dict(meta))
        self.rows.append(row); start=self.clock();settled=False
        try:
            stage['phase']='upstream_open'
            with self.transport(wire,timeout) as up:
                stage['phase']='upstream_read'
                body,usage,models,fps,finish,reason,seen,ids=self._read(up,cap)
            stage['phase']='upstream_verify'
            n,m,r=counts(usage)
            row.update(finish=finish,versions=sorted(fps),response_models=sorted(models),reasoning_stream_seen=seen)
            if n>bound or m>cap:
                # An overrun is a budget discrepancy, not a successful turn or an unmetered retry.
                # The report is retained; quarantine the lease and charge at least its reservation once.
                in_charge,out_charge=max(n,bound),max(m,cap)
                self.spent_input+=in_charge;self.spent_output+=out_charge;settled=True
                row.update(input_tokens=in_charge,output_including_reasoning=out_charge,reasoning_tokens=r,
                           usage='usage_uncertain',reported_input_tokens=n,reported_output_tokens=m,
                           input_overrun=max(0,n-bound),output_overrun=max(0,m-cap))
                raise LeaseError('provider exceeded reservation')
            self.spent_input+=n;self.spent_output+=m;settled=True
            row.update(input_tokens=n,output_including_reasoning=m,reasoning_tokens=r,usage='settled')
            need(models=={MODEL} and (not self.fingerprint or fps=={self.fingerprint}), 'server model or version mismatch')
            need(not (set(ids)&self.seen_ids.keys()),'provider reused tool-call ID')
            if finish=='length':
                # Never let Pi process fragments from a truncated turn, including partial tool arguments.
                # The upstream response has already been fully verified and its usage settled once.
                row['result']=GENERATION_TRUNCATED;self.poisoned=True
                terminal=dict(model=MODEL,choices=[dict(index=0,delta={},finish_reason='length')],usage=usage)
                if len(fps)==1:terminal['system_fingerprint']=next(iter(fps))
                return b'data: '+canonical(terminal)+b'\n\ndata: [DONE]\n\n'
            if finish in {'stop','tool_calls'}:self._record_reasoning(ids,reason)
            # Content filtering is also billed, but is not a successful action.
            row['result']='complete' if finish in {'stop','tool_calls'} else finish
            if finish not in {'stop','tool_calls'}:self.poisoned=True
            return body
        except (ValueError,KeyError,OSError,HTTPException,TypeError,RecursionError,RuntimeError) as e:
            self.poisoned=True
            if not settled:
                self.spent_input+=bound;self.spent_output+=cap
                row.update(usage='uncertain_charged_reservation',input_tokens=bound,output_including_reasoning=cap,reasoning_tokens=None)
            row['result']='usage_uncertain' if row['usage']=='usage_uncertain' else 'transport_or_verification_failed'
            if row['usage']=='usage_uncertain':stage['upstream_reason']='usage_uncertain'
            elif isinstance(e,HTTPError):
                stage['upstream_reason']='upstream_http'
                if type(e.code) is int and 100<=e.code<=599:row['http_status']=stage['http_status']=e.code
                e.close()
            elif isinstance(e,(OSError,HTTPException)):stage['upstream_reason']='upstream_transport'
            elif isinstance(e,LeaseError) and str(e)=='provider wall lease exhausted':stage['upstream_reason']='wall_lease'
            elif isinstance(e,(ValueError,KeyError,TypeError,RecursionError)) and stage['phase']!='upstream_open':
                stage['upstream_reason']='upstream_verification'
            else:stage['upstream_reason']='unknown'
            raise LeaseError('provider stream failed verification; no automatic retry') from None
        finally:
            row['wall_ms'] = round((self.clock()-start)*1000)
            timing = row.get('generation_timing')
            if timing is not None:
                first = timing.pop('first_delta_monotonic')
                last = timing.pop('last_delta_monotonic')
                timing.update(first_delta_ms=round((first-start)*1000),
                              last_delta_ms=round((last-start)*1000),
                              output_span_ms=round((last-first)*1000))
            self._settle_upstream(row)
