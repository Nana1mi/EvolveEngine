"""Select intact WildChat and LongBench-E tasks by rendered Spark token count."""
import argparse
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_chat_timing import W3_INPUT_RANGE, bounds as length_bounds
from experiments.spark26_gguf.level3 import node_preflight as node
from experiments.spark26_gguf.level3.stage_chat_timing import load_model_tokenizer, render_chat

WILDCHAT = 'https://huggingface.co/datasets/allenai/WildChat-1M'
LONGBENCH = 'https://huggingface.co/datasets/zai-org/LongBench'
PERSONAL = re.compile(
    r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|\b(?:\+?\d[\s()-]?){9,15}\b|'
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:sk-|ghp_)[A-Za-z0-9_-]{15,}|'
    r'(?i:\b(?:password|api[_ -]?key|access[_ -]?token|social security|身份证号)\s*[:=])',
    re.IGNORECASE,
)
TASKS = (
    ('code',r'(?i)\b(code|python|sql|javascript|programming|function|debug)\b|编程|代码|函数'),
    ('math',r'(?i)\b(math|equation|calculate|probability|integral|algebra|geometry)\b|数学|方程|概率|计算'),
    ('reasoning',r'(?i)\b(reason|reasoning|compare|deduce|infer|why|explain|analy[sz]e)\b|为什么|分析|推理|比较'),
    ('writing',r'(?i)\b(write|rewrite|draft|compose|story|poem|essay)\b|创作|写一|改写|润色'),
    ('qa',r'\?|？|(?i:\bwhat\b|\bwhere\b|\bhow\b|\bwhen\b)'),
)


def category(text):
    return next((name for name, pattern in TASKS if re.search(pattern,text)), 'other')


def privacy_safe(messages):
    return all(not PERSONAL.search(m['content']) and not re.search(
        r'<\|(?:User|Bot|System|Tool)\|>|<｜(?:start|end)▁of▁sentence｜>|\x00',m['content']) for m in messages)


def wildchat_rows(files, revision):
    import pyarrow.parquet as pq

    for file in sorted(files):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=512,columns=['conversation','conversation_hash','toxic','redacted']):
            for row in batch.to_pylist():
                if row['toxic'] is not False or row['redacted'] is not False:
                    continue
                history = []
                for turn in row['conversation']:
                    role, content = turn.get('role'), turn.get('content')
                    if role not in ('system','user','assistant') or type(content) is not str or not content.strip():
                        break
                    history.append(dict(role=role,content=content))
                    identity = turn.get('turn_identifier')
                    if role == 'user' and (type(identity) is int or type(identity) is str and identity):
                        yield (row['conversation_hash'], dict(messages=history.copy(),source=WILDCHAT+'@'+revision,
                            source_id=str(identity),license='ODC-BY-1.0',task=category(content)))


def longbench_rows(file, revision, official, licenses):
    with zipfile.ZipFile(file) as archive:
        for name in sorted(archive.namelist()):
            path = Path(name)
            if not path.name.endswith('_e.jsonl'):
                continue
            task = path.name.removesuffix('_e.jsonl')
            if task not in official or (licenses is not None and task not in licenses):
                continue
            pattern = official[task]
            # Preserve the original task and full context, without targets, adaptation or padding.
            if '{context}' not in pattern:
                continue
            with archive.open(name) as source:
                for line in source:
                    row = json.loads(line)
                    context = row['context']
                    original = pattern.format(context=context,input=row['input'])
                    yield (digest(context.encode('utf-8')),dict(messages=[dict(role='user',content=original)],
                           source=LONGBENCH+'@'+revision,source_id=task+'_e/'+row['_id'],
                           license=licenses[task] if licenses is not None else 'unreviewed',task=task))


def intake(rows, seeds, lengths, tokenizer, template, *, source_name):
    pools = defaultdict(list)
    long_groups = defaultdict(list)
    seen_messages = set()
    overlong_groups = set()
    statistics = Counter()
    for group,row in rows:
        statistics['eligible_source_rows'] += 1
        messages = row['messages']
        if group in overlong_groups:
            continue
        if not type(group) is str or not group or not privacy_safe(messages):
            statistics['privacy_or_format_rejected'] += 1
            continue
        text = render_chat(tokenizer,template,messages)
        ids = tokenizer.encode(text,add_special_tokens=False)
        statistics['rendered_rows'] += 1
        length = 4096 if source_name == 'longbench' and W3_INPUT_RANGE[0] <= len(ids) <= W3_INPUT_RANGE[1] else len(ids)
        bucket = int(digest(canonical([seeds['partition'],group]))[:8],16) % 5
        split = 'generalization' if bucket==0 else 'holdout' if bucket in (1,2) else 'development'
        stratum=None
        if source_name=='wildchat' and split=='generalization':
            stratum=next(((n,b) for n in sorted(lengths) for b in (0,1)
                          if length_bounds('',n,split,b)[0]<=len(ids)<=length_bounds('',n,split,b)[1]),None)
        if length not in lengths and stratum is None:
            statistics['length_rejected'] += 1
            if source_name=='wildchat' and length>max(lengths)*(1.25 if split=='generalization' else 1):
                # Later turns append complete role-delimited turns to this history.
                overlong_groups.add(group)
            continue
        pin = digest(canonical(ids))
        if pin in seen_messages:
            statistics['duplicate_rendered'] += 1
            continue
        seen_messages.add(pin)
        rank = digest(canonical([seeds['order'],row['source'],row['source_id']]))
        if source_name == 'longbench':
            long_groups[group].append((rank,row,len(ids)))
            statistics[f'w3_input_tokens:{len(ids)}'] += 1
            continue
        if split=='generalization':
            if stratum is None: continue
            length=stratum
        pools[split,length].append((rank,row))
        statistics[f'{split}:{length}:{row["task"]}'] += 1
    if long_groups:
        targets = {'development':2,'holdout':2,'generalization':1}
        occupancy = {split:Counter() for split in targets}
        ordered = sorted(long_groups,key=lambda key:(-len(long_groups[key]),
                          digest(canonical([seeds['partition'],key]))))
        for group in ordered:
            group_rows = long_groups[group]
            strata = Counter((row['task'],n//64) for _,row,n in group_rows)
            split = min(targets,key=lambda s:(
                sum(strata[k]*occupancy[s][k] for k in strata)/targets[s],
                (sum(occupancy[s].values())+len(group_rows))/targets[s],
                digest(canonical([seeds['partition'],group,s]))))
            occupancy[split].update(strata)
            for rank,row,n in group_rows:
                if split=='generalization' and n==4096: continue
                key=(4096,int(n>4096)) if split=='generalization' else 4096
                pools[split,key].append((rank,row))
                statistics[f'{split}:4096:{row["task"]}'] += 1
                statistics[f'{split}:w3_input_tokens:{n}'] += 1
    return pools,dict(statistics)


def allocate(raw_pools, shapes, desired_waves, minimum):
    pools = {}
    for key, rows in raw_pools.items():
        categories = defaultdict(deque)
        for _,row in sorted(rows,key=lambda x:x[0]):
            categories[row['task']].append(row)
        ordered = deque()
        while any(categories.values()):
            for queue in categories.values():
                if queue:
                    ordered.append(queue.popleft())
        pools[key] = ordered
    lacks = {}
    wave_totals = {}
    for split,waves in desired_waves.items():
        if split=='generalization':
            demand=Counter(n for shape in shapes.values() for n,_ in shape)
            for n,count in demand.items():
                for b in (0,1):
                    want=count*((minimum+1-b)//2); found=len(pools.get((split,(n,b)),()))
                    if found<want: lacks[f'{split}:{n}:{b}']=dict(wanted=want,found=found)
            wave_totals[split]=minimum
            continue
        demand = Counter(n for scene,shape in shapes.items() if scene!='W3' for n,_ in shape)
        for n, count in demand.items():
            actual = len(pools.get((split,n),[]))
            if actual < minimum*count:
                lacks[f'{split}:{n}'] = dict(wanted=minimum*count, found=actual)
        w3 = len(pools.get((split,4096),[]))
        if w3 < minimum:
            lacks[f'{split}:4096'] = dict(wanted=minimum,found=w3)
        wave_totals[split] = min(waves,*(len(pools.get((split,n),[]))//count for n,count in demand.items()))
    if lacks:
        return None,lacks
    out = {}
    for split,waves in wave_totals.items():
        out[split] = {scene:[] for scene in shapes}
        if split=='generalization':
            for wave in range(waves):
                for scene,shape in shapes.items(): out[split][scene].append([pools[split,(n,wave%2)].popleft() for n,_ in shape])
            continue
        for _ in range(waves):
            for scene,shape in shapes.items():
                if scene != 'W3':
                    out[split][scene].append([pools[split,n].popleft() for n,_ in shape])
        out[split]['W3'] = [[row] for row in pools[split,4096]]
    return out,{}


def file_digest(path):
    with path.open('rb') as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    h = sha256()
    for block in iter(lambda:stream.read(1<<20),b''):
        h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--wildchat',type=Path,nargs='*',default=[])
    parser.add_argument('--wildchat-revision',required=True)
    parser.add_argument('--longbench',type=Path,required=True)
    parser.add_argument('--longbench-revision',required=True)
    parser.add_argument('--longbench-prompts',type=Path,required=True)
    parser.add_argument('--longbench-prompt-revision',required=True)
    parser.add_argument('--longbench-license',action='append',default=[],metavar='SUBSET=LICENSE')
    parser.add_argument('--seed-file',type=Path,required=True)
    parser.add_argument('--scan-only',action='store_true')
    args = parser.parse_args()
    seeds = json.loads(args.seed_file.read_bytes())
    node.need(set(seeds)=={'partition','order'} and all(re.fullmatch('[0-9a-f]{64}',seed) for seed in seeds.values()),
              'require private pinned split and ordering seeds')
    codec,template,pins = load_model_tokenizer(args.model)
    _,plan = node.collector_pins()
    shapes = plan['scenes']
    lengths = {n for scene,shape in shapes.items() if scene!='W3' for n,_ in shape}
    docs = json.loads(args.longbench_prompts.read_bytes())
    licenses = dict(item.split('=',1) for item in args.longbench_license)
    long, stats = intake(longbench_rows(args.longbench,args.longbench_revision,docs,licenses or None),
                         seeds,{shapes['W3'][0][0]},codec,template,source_name='longbench')
    print('longbench',json.dumps(stats,sort_keys=True),flush=True)
    wild, wild_stats = intake(wildchat_rows(args.wildchat,args.wildchat_revision),seeds,lengths,
                              codec,template,source_name='wildchat')
    print('wildchat',json.dumps(wild_stats,sort_keys=True),flush=True)
    required = 2*plan['warm_waves']+plan['timed_waves']
    desired = {'development':5*required,'holdout':5*required,'generalization':required}
    selected, missing = allocate(long | wild, shapes, desired, required)
    metadata = dict(seed_commitment=digest(canonical(seeds)),tokenizer=pins['tokenizer.json'],
                    template=pins['chat_template.jinja'],enable_thinking=True,
                    wildchat=dict(revision=args.wildchat_revision,shards={f.name:file_digest(f) for f in args.wildchat},
                                  statistics=wild_stats),
                    longbench=dict(revision=args.longbench_revision,archive_sha256=file_digest(args.longbench),
                                   prompt_revision=args.longbench_prompt_revision,
                                   prompt_sha256=file_digest(args.longbench_prompts), statistics=stats,licenses=licenses),
                    missing=missing)
    report = args.output.with_suffix('.intake.json')
    with report.open('xb') as out:
        out.write(canonical(metadata)+b'\n')
    print('missing',json.dumps(missing,sort_keys=True),flush=True)
    if args.scan_only:
        return
    node.need(licenses and selected is not None,'public data could not supply the frozen request pools')
    with args.output.open('xb') as out:
        out.write(canonical(dict(schema='public-chat-task-pools-v1',pools=selected,metadata=metadata))+b'\n')


if __name__=='__main__':
    main()
