"""Frozen development trajectory protocol. No model imports and no candidate code.

Candidate generates its own continuations. The independent teacher only receives
host-reconstructed token prefixes, never candidate probabilities or a candidate model path.
"""
from array import array
import base64
from hashlib import sha256
import json
import math
import random
import re
import sys
import zlib

VERSION = 'paired-isolation-teacher-prefill-v2'
VOCAB, EOS, STEPS = 131072, 1, 5
CASES = (0, 9)
TRIGGERS = (8, 32)
SURVIVOR, NEIGHBOR = 160, 128
MARKERS = (1, 4, 42, 220, 2003, 6510, 7727, 16081, 28000, 97001)
TAIL = (SURVIVOR-2, SURVIVOR-1)


def need(ok, msg):
    if not ok: raise ValueError(msg)


def prefix(case):
    # Registered DEV synthetic inputs, independent of all scoring/holdout prompts.
    rng = random.Random(0x20261123 + case)
    alphabet = [rng.randrange(2000,28000) for _ in range(7+case%19)]
    return [rng.choice(alphabet) for _ in range((130,260)[case%2])]


def ih(ids):
    return sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def token(t): return type(t) is int and 0 <= t < VOCAB


def lp(x): return type(x) in (int,float) and math.isfinite(x) and x <= 0


def request_id(rid): return isinstance(rid,str) and re.fullmatch(r'[0-9a-f]{32}',rid) is not None


def pack(pairs):
    need(type(pairs) is list and len(pairs) == VOCAB, 'full vocabulary required')
    v, seen = array('f',[0])*VOCAB, bytearray(VOCAB)
    for value, t, *_ in pairs:
        need(token(t) and not seen[t] and lp(value), 'duplicated token or invalid logprob')
        seen[t], v[t] = 1, value
    if sys.byteorder != 'little': v.byteswap()
    raw = v.tobytes()
    return dict(sha256=sha256(raw).hexdigest(), f32le_zlib=base64.b64encode(zlib.compress(raw)).decode('ascii'))


def unpack(row):
    need(type(row) is dict and set(row) == {'sha256','f32le_zlib'} and isinstance(row['f32le_zlib'],str)
         and len(row['f32le_zlib']) <= 8*VOCAB, 'invalid vocabulary frame')
    d = zlib.decompressobj()
    raw = d.decompress(base64.b64decode(row['f32le_zlib'],validate=True),4*VOCAB+1)
    need(d.eof and not d.unused_data and len(raw)==4*VOCAB and sha256(raw).hexdigest()==row['sha256'],
         'invalid vocabulary size or digest')
    v = array('f'); v.frombytes(raw)
    if sys.byteorder != 'little': v.byteswap()
    need(all(math.isfinite(x) and x <= 0 for x in v) and abs(math.fsum(map(math.exp,v))-1)<.001,
         'non-normalized full-vocabulary logprobs')
    return v


def check_decode(row, case):
    need(row['case']==case and request_id(row['rid']) and row['prompt_sha']==ih(prefix(case))
         and row['input']==len(prefix(case)) and row['cap']==len(row['ids'])==len(row['vectors'])==STEPS and all(map(token,row['ids']))
         and row['finish']=='length','incomplete autonomous decode')


def streams(pair, trigger):
    """Verify overlapping requests and a real, acknowledged cancellation, not merely an abort RPC."""
    need(pair['trigger']==trigger,'wrong cancellation point')
    for arm in ('complete','cancel'):
        s,t = pair[arm]
        for x,base,cap in ((s,7,SURVIVOR),(t,6,NEIGHBOR)):
            ids, progress = x['ids'], x['progress']
            need(request_id(x['rid']) and x['prompt_sha']==ih(prefix(base)) and x['cap']==cap and x['done'] is True
                 and ids and len(ids)<=cap and all(map(token,ids)) and progress and progress[-1][0]==len(ids)
                 and all(type(n) is int and 0<=n<=cap and type(ts) is int and ts>0 for n,ts in progress)
                 and all(a<=b and c<=d for (a,c),(b,d) in zip(progress,progress[1:])), 'unbound mixed stream')
        need(s['finish']=='length' and len(s['ids'])==SURVIVOR and s['ack'] is None and
             len(s['probs'])==SURVIVOR and all(len(row)==len(MARKERS)+1 and all(map(lp,row)) for row in s['probs']),
             'incomplete survivor probabilities')
        def active(x): return [ts for n,ts in x['progress'] if n>0]
        st, tt = active(s),active(t)
        need(st and tt and max(st[0],tt[0])<min(st[-1],tt[-1]),'no overlapping decode witness')
        if arm=='complete':
            need(t['ack'] is None and (t['finish']=='length' and len(t['ids'])==NEIGHBOR or
                 t['finish']=='stop' and t['ids'][-1]==EOS and EOS not in t['ids'][:-1]), 'neighbor did not complete normally')
        else:
            n, start, ack, http = t['ack']
            need(http==200 and t['finish']=='abort' and trigger<=n<=len(t['ids'])<NEIGHBOR and start<=ack
                 and any(k==n and when<=start for k,when in t['progress']) and
                 any(0<k<SURVIVOR and when<start for k,when in s['progress']) and
                 any(k<=TAIL[0] and when>ack for k,when in s['progress']), 'unwitnessed in-flight cancellation')
    return pair


def teacher_plan(candidate):
    need(candidate['protocol']==VERSION and candidate['role']=='candidate' and len(candidate['decode'])==len(CASES)
         and len(candidate['isolation'])==len(TRIGGERS),'incomplete candidate trajectory')
    work, rids = [], []
    def add(key, ids, wanted):
        work.append(dict(key=key, ids=ids, wanted=wanted))
    for row, case in zip(candidate['decode'],CASES):
        check_decode(row,case); rids.append(row['rid'])
        for i in range(STEPS): add(f'd/{case}/{i}',prefix(case)+row['ids'][:i], 'all')
    for pair,trigger in zip(candidate['isolation'],TRIGGERS):
        streams(pair,trigger)
        for arm in ('complete','cancel'):
            s,t = pair[arm]; rids.extend((s['rid'],t['rid']))
            for i in TAIL:
                add(f'i/{trigger}/{arm}/{i}',prefix(7)+s['ids'][:i],sorted(set((*MARKERS,s['ids'][i]))))
    need(len(set(rids))==len(rids),'reused candidate request ID')
    return work


def compare(candidate, teacher):
    plan = teacher_plan(candidate)
    need(teacher['role']=='teacher' and teacher['protocol']==VERSION and len(teacher['rows'])==len(plan), 'incomplete E0 teacher')
    seen = {r['rid'] for r in candidate['decode']} | {r['rid'] for p in candidate['isolation'] for a in ('complete','cancel') for r in p[a]}
    peak, quantum, observations = {}, {}, []
    def ulp(x): return math.ldexp(1.,max(-126,math.frexp(abs(x))[1]-1)-7)
    def noise(k,v): quantum[k]=max(quantum.get(k,0.),v)
    def record(metric, value): peak[metric] = max(peak.get(metric,0.), value)
    lookup = {p['case']:p for p in candidate['decode']}; iso = {p['trigger']:p for p in candidate['isolation']}
    residual = {}
    for job, row in zip(plan,teacher['rows']):
        need(request_id(row['rid']) and row['rid'] not in seen and row['key']==job['key'] and
             row['prompt_sha']==ih(job['ids']) and row['input']==len(job['ids']) and row['cap']==1
             and row['finish']=='length' and token(row['generated']), 'teacher request/prefix differs')
        seen.add(row['rid']); fields = job['key'].split('/'); i = int(fields[-1])
        if job['wanted']=='all':
            c = lookup[int(fields[1])]; a,b=unpack(c['vectors'][i]),unpack(row['vector'])
            tv=math.fsum(abs(math.exp(x)-math.exp(y)) for x,y in zip(a,b))/2
            eos=abs(a[EOS]-b[EOS]); regret=max(b)-b[c['ids'][i]]
            # One BF16 representable increment in the E0 logprobs at these actual prefixes.
            noise('decode_tv',math.fsum(math.exp(x)*ulp(x) for x in b)/2)
            noise('decode_eos_logp',ulp(b[EOS])); noise('decode_greedy_regret',ulp(max(b)))
            record('decode_tv',tv); record('decode_eos_logp',eos); record('decode_greedy_regret',regret)
            observations.append(dict(key=job['key'],path='first_prefill' if i==0 else 'autoregressive_step',tv=tv,eos_logprob_delta=eos,regret=regret))
        else:
            _,stage,arm,_=fields; s=iso[int(stage)][arm][0]; probs=s['probs'][i]
            ids=(*MARKERS,s['ids'][i]); vals=row['values']
            need(type(vals) is list and len(vals)==len(job['wanted']) and all(map(lp,vals)), 'teacher slice differs')
            ref=dict(zip(job['wanted'],vals)); delta=[math.exp(x)-math.exp(ref[t]) for t,x in zip(ids,probs)]
            quantum_prob=max(math.exp(v)*ulp(v) for v in ref.values())
            noise('iso_probability',quantum_prob); noise('iso_eos_logp',ulp(ref[EOS])); noise('iso_paired_residual',2*quantum_prob)
            residual[stage,arm,i]=delta[:-1]; record('iso_probability',max(map(abs,delta)))
            record('iso_eos_logp',abs(probs[0]-ref[EOS]))
    for stage in TRIGGERS:
        for i in TAIL:
            a,b=(residual[str(stage),arm,i] for arm in ('complete','cancel'))
            record('iso_paired_residual',max(abs(x-y) for x,y in zip(a,b)))
    return dict(protocol=VERSION,maxima=peak,reference_quantum=quantum,decode_positions=observations,isolation_post_cancel_positions=len(TRIGGERS)*len(TAIL))


METRICS = {'decode_tv','decode_eos_logp','decode_greedy_regret','iso_probability','iso_eos_logp','iso_paired_residual'}


def envelope(reports):
    # Engineering envelope, not a confidence interval. Two independently deployed E0 arms are required
    # by the calibration producer. Register before looking at any candidate data or held-out inputs.
    need(len(reports)==2,'two independent E0 measurements required')
    for report in reports:
        need(report['protocol']==VERSION and all(set(report[k])==METRICS and all(type(v) in (int,float) and
             math.isfinite(v) and v>=0 for v in report[k].values()) for k in ('maxima','reference_quantum')),
             'incomplete E0 metric calibration')
    out={k:max(max(2*r['maxima'][k],r['reference_quantum'][k]) for r in reports) for k in sorted(METRICS)}
    need(max(r['maxima']['decode_tv'] for r in reports)<=.05,'observed E0 TV exceeds DEV ceiling; no tolerance issued')
    out['decode_tv']=min(.05,out['decode_tv'])  # safety margin must never relax the absolute DEV ceiling
    return out


def verdict(report, limits):
    need(report['protocol']==VERSION and set(limits)==set(report['maxima'])==METRICS and
         all(type(v) in (int,float) and math.isfinite(v) and v>=0 for d in (limits,report['maxima']) for v in d.values()),
         'no registered trajectory verdict')
    failed=sorted(k for k in limits if report['maxima'][k]>limits[k])
    return dict(status='violation' if failed else 'valid',failed=failed,maxima=report['maxima'],protocol=VERSION)
