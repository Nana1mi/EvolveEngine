"""Extract a bounded, redacted runtime fingerprint from saved isolated SGLang logs.

This is a partial projection, not an identity attestation or authorization to run.
Parse only already authorized local copies; never read an online endpoint.
"""
import argparse
import ast
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

MAX_LOG = 32 << 20
MAX_LINE = 1 << 17
SCHEMA = 1
SCOPE = 'allowlisted_sglang_runtime_projection_not_production_attestation'
CATEGORICAL = re.compile(r'[A-Za-z0-9_+\-]{1,48}\Z')

# Exact pinned field names; exclude paths, addresses, model IDs, auth and all raw environment.
FIELD_TYPES = {
    'load_format':str, 'dtype':str, 'kv_cache_dtype':str, 'attention_backend':str,
    'prefill_attention_backend':str, 'decode_attention_backend':str,
    'cuda_graph_backend_decode':str, 'cuda_graph_backend_prefill':str,
    'disable_cuda_graph':bool, 'disable_decode_cuda_graph':bool, 'disable_prefill_cuda_graph':bool,
    'disable_radix_cache':bool, 'disable_chunked_prefix_cache':bool,
    'tp_size':int, 'pp_size':int, 'dp_size':int, 'attn_dp_size':int, 'attn_cp_size':int,
    'max_running_requests':int, 'context_length':int, 'max_total_tokens':int,
    'chunked_prefill_size':int, 'max_prefill_tokens':int, 'mem_fraction_static':float,
    'page_size':int, 'enable_mixed_chunk':bool, 'enable_dynamic_chunking':bool,
    'schedule_policy':str, 'disable_overlap_schedule':bool,
    'enable_deterministic_inference':bool, 'stream_interval':int,
    'incremental_streaming_output':bool, 'tool_call_parser':str, 'reasoning_parser':str,
    'speculative_algorithm':str, 'speculative_num_steps':int, 'speculative_eagle_topk':int,
    'speculative_num_draft_tokens':int, 'disaggregation_mode':str,
}
SCHED_NAMES = ('max_total_num_tokens','chunked_prefill_size','max_prefill_tokens','max_running_requests','context_len')
SCHEDULER = re.compile(', '.join(fr'{s}=(-?\d+)' for s in SCHED_NAMES) + r'(?:,|\s*$)')
KV = re.compile(r'\b(Full|SWA) KV Cache is allocated\. dtype: torch\.([a-z0-9_]+), #tokens: (\d+),')


def make_value(key,value):
    want = FIELD_TYPES[key]
    if type(value) is not want and not (want is float and type(value) is int):
        return None, 'unexpected_type_or_unset'
    if want is str and not CATEGORICAL.fullmatch(value):
        return None, 'noncategorical_value_omitted'
    if want is float and not isfinite(value):
        return None, 'nonfinite_value_omitted'
    return value, None


def project(raw):
    if not 0 < len(raw) <= MAX_LOG:
        raise ValueError('Runtime log is empty or exceeds the bounded size')
    lines = raw.decode('utf-8').splitlines()
    # Do not concatenate final logs from different boots or accept a CLI command as evidence.
    starts = [line.split('server_args=',1)[1] for line in lines if '] server_args={' in line]
    if len(starts) != 1 or len(starts[0]) > MAX_LINE:
        raise ValueError('Need exactly one bounded server_args record from a startup')
    record = ast.parse(starts[0], mode='eval').body
    if not isinstance(record,ast.Dict):
        raise ValueError('Expected a server_args dictionary')
    keys = [ast.literal_eval(k) for k in record.keys]
    if any(not isinstance(k,str) for k in keys) or len(keys) != len(set(keys)):
        raise ValueError('server_args keys must be unique strings')

    # Only literal-evaluate selected values; unprinted fields are not exported even if present.
    settings, unavailable = {}, {}
    for key, val in zip(keys,record.values,strict=True):
        if key not in FIELD_TYPES:
            continue
        value = ast.literal_eval(val)
        if value is None:
            # Null may mean inherited/auto or explicitly unused. Keep it distinct from omission.
            settings[key] = None
            continue
        setting, why = make_value(key,value)
        if why:
            unavailable[key] = why
        else:
            settings[key] = setting
    for k in sorted(set(FIELD_TYPES)-settings.keys()-unavailable.keys()):
        unavailable[k] = 'not_present_in_log'

    effective = {tuple(map(int,m.groups())) for line in lines if (m:=SCHEDULER.search(line))}
    if len(effective) > 1:
        raise ValueError('Conflicting runtime scheduler capacity rows; resolve instance/rank before comparing')
    caps = dict(zip(SCHED_NAMES, next(iter(effective)),strict=True)) if effective else {}
    pool_rows = {}
    for line in lines:
        if m:=KV.search(line):
            name,dtype,tokens=m.groups()
            value=(dtype,int(tokens))
            if name in pool_rows and pool_rows[name]!=value:
                raise ValueError('Conflicting data for one runtime KV pool')
            pool_rows[name]=value
    return dict(schema=SCHEMA, scope=SCOPE, source_log_sha256=sha256(raw).hexdigest(),
                startup_args=settings, unavailable_startup_args=unavailable,
                scheduler_capacity=caps, kv_pools={k.lower():dict(dtype=d,tokens=n) for k,(d,n) in pool_rows.items()})


def compare(a,b):
    if any(doc.get('schema')!=SCHEMA or doc.get('scope')!=SCOPE for doc in (a,b)):
        raise ValueError('Incompatible runtime projection format')
    gaps, changes, equal = [], [], []
    for group in ('startup_args','scheduler_capacity','kv_pools'):
        x,y=a[group],b[group]
        covered = set(x)|set(y)
        if group=='startup_args':
            covered |= FIELD_TYPES.keys()
        if group=='scheduler_capacity':
            covered |= set(SCHED_NAMES)
        if group=='kv_pools' and not covered:
            gaps.append('kv_pools')
        for k in sorted(covered):
            path=group+'.'+k
            if k not in x or k not in y:
                gaps.append(path)
            elif x[k]!=y[k] or type(x[k]) is not type(y[k]):
                changes.append(dict(field=path, left=x[k],right=y[k]))
            else:
                equal.append(path)
    return dict(equal_fields=equal, differing_fields=changes, unobserved_fields=gaps,
                null_or_auto_startup_args=sorted(k for k in FIELD_TYPES if any(
                    k in d['startup_args'] and d['startup_args'][k] in (None,'auto') for d in (a,b))),
                comparison='partial_only', production_authorized=False)


def read_projection(path):
    with path.open('rb') as stream:
        return project(stream.read(MAX_LOG+1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    p=parser.add_subparsers(dest='verb',required=True)
    p.add_parser('project').add_argument('saved_log',type=Path)
    p_cmp=p.add_parser('compare')
    p_cmp.add_argument('left_log',type=Path)
    p_cmp.add_argument('right_log',type=Path)
    args=parser.parse_args()
    if args.verb=='project':
        result=read_projection(args.saved_log)
    else:
        result=compare(read_projection(args.left_log),read_projection(args.right_log))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
