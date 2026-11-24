"""Prespecified G1-A real-GB10 cases; never runs a free Agent session."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from node_client import ALLOWED, PLAN, bundle_manifest, connection, invoke, need
from audit_worker import audit
from verify_common import main as integrated_worker


# Real BF16 full-matrix copy of the QKV weight of every layer, computed from
# Q8_0 on the actual path, retained after requests, and deliberately not used.
# Thus a pure numerical gate should pass the fault.
ANCHOR = '        shard_id = layer.qweight.shard_id\n\n        if shard_id:\n'
COPY = '''        shard_id = layer.qweight.shard_id

        if (type(layer).__name__ == "QKVParallelLinear"
                and not hasattr(layer, "_fixture_bf16_weight_copy")):
            layer._fixture_bf16_weight_copy = dequantize_gguf_weight(
                layer.qweight, WeightType.Q8_0, torch.bfloat16
            )

        if shard_id:
'''


def fault(src):
    text=src.decode();need(text.count(ANCHOR)==1,'missing fault placement');out=text.replace(ANCHOR,COPY).encode()
    compile(out,'gguf.py','exec')
    return out


def run(conn,source,archive,j):
    raw=source.read_bytes();need(sha256(raw).hexdigest()==PLAN['gguf_py'],'unexpected E0')
    path=ALLOWED[0];base={path:raw};j['bundle_sha']=sha256(bundle_manifest()).hexdigest();j['cases']={}
    missing_checked=False
    for case,edits,expected in (('e0',{},'valid'),('retained_qkv_bf16',{path:fault(raw)},'violation')):
        h=sha256(edits.get(path,raw)).hexdigest();row=j['cases'][case]=dict(source_sha=h)
        root,rc=invoke(conn,'develop_quality',edits,h,base,200000);row.update(archive=str(root),rc=rc)
        need(rc==0,'worker failed to collect complete evidence')
        q=audit(root,archive/'spark26-v32-g1-d3.lv2uVT',archive/'g1-dev-fJPjW8');row['result']=q
        need(q['status']==expected and q['max_tv']==0 and q['eos_passed']==4 and q['mixed_passed'], 'wrong gate outcome')
        if case=='e0':
            from audit_worker import residency,raw_rows
            rows=raw_rows(root/'probe.jsonl')
            try:residency(rows[:-1],(root/'service.log').read_text(encoding='utf8'),root)
            except ValueError as exc:
                need(str(exc)=='incomplete external residency stages','unexpected test failure');missing_checked=True
        else:
            need(q['residency']['extra_bytes']==28*3072*2048*2,'wrong BF16 mirror coverage')
    need(missing_checked,'missing external phase was not blocked');j['missing_phase']='insufficient_evidence'
    j['common_runner']={};integrated_worker(conn,source,archive,j['common_runner'])
    q,m=j['common_runner']['worker'];need(q['result']['residency']['status']=='within_observed_scope' and m['result']['status']=='observed','positive end-to-end evidence missing')
    j['status']='pass_g1a_pre_pilot_scope'


if __name__=='__main__':
    j=dict(status='not_passed',scope='quality-only residence across stages, packed CUDA inventory and torch allocator, actual QKV BF16 mirror negative, no free Agent')
    with Path(sys.argv[2]).open('x',encoding='utf8') as f:
        try:
            with connection() as conn:run(conn,Path(sys.argv[1]),Path(os.environ['LOCALAPPDATA'])/'EvolveEngine',j)
        except (OSError,ValueError,RuntimeError,TypeError,KeyError) as e:j['failure']=f'{type(e).__name__}: {e}'
        finally:f.write(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
    print(j['status'],j.get('failure',''));raise SystemExit(j['status']=='not_passed')
