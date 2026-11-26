"""Prepare and independently screen thinking-enabled Level 3 chat requests."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from secrets import token_hex
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_chat_timing import SCHEMA, SPLITS, W3_INPUT_RANGE, load_chat
from evolveengine.v3_level3_collect import stage_inputs, write
from evolveengine.v3_level3_workload import MAX_CORPUS_BYTES, input_fingerprints
from experiments.spark26_gguf.level3 import node_preflight as node, replay_e0
from experiments.spark26_gguf.level3.stage_timing_corpus import pinned


def load_model_tokenizer(model):
    from transformers import PreTrainedTokenizerFast

    pins = dict(tuple(reversed(line.split('  ',1))) for line in (node.DEV/'model.sha256').read_text().splitlines())
    codec_path = model/'tokenizer.json'
    pinned(codec_path,pins['tokenizer.json'],MAX_CORPUS_BYTES)
    template = pinned(model/'chat_template.jinja',pins['chat_template.jinja'],1<<20).decode('utf-8')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(codec_path), chat_template=template)
    return tokenizer,template,pins


def render_chat(tokenizer, template, messages, *, thinking=True):
    return tokenizer.apply_chat_template(messages,chat_template=template,tokenize=False,
                                         add_generation_prompt=True,enable_thinking=thinking)


def diverse_waves(waves, count):
    available = list(waves)
    picked = []
    categories = Counter()
    while available and len(picked) < count:
        index = min(range(len(available)),key=lambda i:sum(categories[row['task']] for row in available[i]))
        wave = available.pop(index)
        picked.append(wave)
        categories.update(row['task'] for row in wave)
    return picked


def screen_split(source, folder, timing, live, deadline, plan, required):
    witnesses, selected = {}, {}
    for scene,waves in source.items():
        jobs = {scene:[[row['ids'] for row in wave] for wave in waves]}
        planned = dict(plan,scenes={scene:plan['scenes'][scene]},warm_waves=0,
                       timed_waves=len(waves),timing_mode='natural_eos')
        if scene == 'W3':
            planned['w3_input_range'] = W3_INPUT_RANGE
        observed = folder/scene
        observed.mkdir(mode=0o700)
        traffic = observed/'inputs'
        stage_inputs(traffic,{**timing.files,'workload.json':canonical(planned),'inputs.json':canonical(jobs)})
        raw = timing.observer(live,traffic,observed,deadline,script='observe_text_timing.py',output_cap=4<<20)
        reports = list(map(json.loads,raw.splitlines()))
        pins = input_fingerprints(jobs)[scene]
        node.need(len(reports)==len(waves),'incomplete E0 screen')
        chosen = []
        rids = set()
        for i,(wave,result) in enumerate(zip(waves,reports)):
            rows = result['requests']
            node.need((result['scene'],result['index'],result['timed'])==(scene,i,True) and
                      tuple(r['input_sha256'] for r in rows)==pins[i] and
                      all(r['stream_complete'] and r['rid'] not in rids for r in rows) and
                      len({r['rid'] for r in rows})==len(rows), 'unbound E0 screen result')
            rids.update(r['rid'] for r in rows)
            if scene=='W3' or all((r['finish_reason'],r['output'],r['output_cap'])==('length',out,out)
                                   for r,(_,out) in zip(rows,plan['scenes'][scene])):
                chosen.append(wave)
        selected[scene] = chosen if scene=='W3' else diverse_waves(chosen,required)
        witnesses[scene] = dict(stdout_sha256=digest(raw),service=live.container_id,waves=len(waves),
                                passed_waves=len(chosen),input_tasks=Counter(row['task'] for wave in waves for row in wave),
                                selected_tasks=Counter(row['task'] for wave in selected[scene] for row in wave))
        write(observed/'screen-summary.json',witnesses[scene])
        print(f'{folder.name} {scene} E0: {len(chosen)} qualifying waves out of {len(waves)}',flush=True)
    insufficient = {s:len(waves) for s,waves in selected.items() if len(waves)<required}
    node.need(not insufficient,f'{folder.name} needs {required} fully screened waves per scene: {insufficient}')
    return selected,witnesses


def prepare(path, pin, model, registration):
    tokenizer,template,pins = load_model_tokenizer(model)
    _, plan = node.collector_pins(registration['timing_text'])
    doc = json.loads(pinned(path,pin,MAX_CORPUS_BYTES))
    node.need(type(doc) is dict and set(doc) == {'schema','pools','metadata'} and doc['schema'] == 'public-chat-task-pools-v1'
              and doc['pools'].keys() == SPLITS, 'expected three separate public task pools')
    node.need(doc['metadata']['enable_thinking'] is True and
              doc['metadata']['tokenizer'] == pins['tokenizer.json'] and
              doc['metadata']['template'] == pins['chat_template.jinja'], 'intake used a different tokenizer or template')
    required = 2*plan['warm_waves']+plan['timed_waves']
    token_pins, source_ids = set(), set()
    pools = {}
    for split, scenes in doc['pools'].items():
        node.need(scenes.keys() == plan['scenes'].keys(), 'public task scene coverage differs')
        pools[split] = {}
        for scene, waves in scenes.items():
            shape = plan['scenes'][scene]
            node.need(type(waves) is list and len(waves) >= required, 'insufficient number of task waves')
            pools[split][scene] = []
            for index,wave in enumerate(waves):
                node.need(type(wave) is list and len(wave) == len(shape), 'unexpected concurrent task count')
                rows = []
                for row,(n,_) in zip(wave,shape):
                    node.need(type(row) is dict and set(row) == {'messages','source','source_id','license','task'} and
                              all(type(row[k]) is str and 0 < len(row[k]) <= 512 for k in ('source','source_id','license','task')),
                              'missing source and license')
                    messages = row['messages']
                    node.need(type(messages) is list and messages and messages[-1].get('role') == 'user'
                              and all(type(m) is dict and set(m) == {'role','content'} and
                                      m['role'] in ('system','user','assistant') and type(m['content']) is str
                                      and m['content'].strip() for m in messages), 'invalid public chat messages')
                    text = render_chat(tokenizer,template,messages)
                    without_thinking = render_chat(tokenizer,template,messages,thinking=False)
                    node.need(text != without_thinking,'the target chat template did not enable thinking')
                    ids = tokenizer.encode(text,add_special_tokens=False)
                    from evolveengine.v3_level3_chat_timing import bounds
                    lo,hi=bounds(scene,n,split,index)
                    size_matches=lo<=len(ids)<=hi
                    node.need(size_matches,f'{split}/{scene} has unexpected prompt length {len(ids)}')
                    key = (row['source'],row['source_id'])
                    token_pin = digest(canonical(ids))
                    node.need(key not in source_ids and token_pin not in token_pins,
                              'request sources and exact prompts must be disjoint')
                    source_ids.add(key)
                    token_pins.add(token_pin)
                    rows.append(dict(ids=ids, **{k:row[k] for k in ('source','source_id','license','task')}))
                pools[split][scene].append(rows)
    return dict(tokenizer_sha256=pins['tokenizer.json'],template_sha256=pins['chat_template.jinja'],
                pools=pools,metadata=doc['metadata'])


def freeze(root, settings, registration, inputs, inputs_pin, e0_build, e0_pin, prior, prior_pin, *, prepared=False):
    from evolveengine.v3_level3_deploy import stage_deployment
    from experiments.spark26_gguf.level3.control_run import read_pinned

    if prepared:
        task = json.loads(pinned(inputs,inputs_pin,MAX_CORPUS_BYTES))
        node.need(task['pools'].keys()==SPLITS and task['metadata']['enable_thinking'] is True and
                  task['metadata']['tokenizer']==task['tokenizer_sha256'] and
                  task['metadata']['template']==task['template_sha256'], 'unbound prepared task manifest')
    else:
        task = prepare(inputs,inputs_pin,Path(settings['model_dir']),settings)
    write(root/'intake.json',dict(staged_manifest_sha256=inputs_pin, **task['metadata']))
    old = read_pinned(prior,prior_pin)
    replay_e0.check_build_compatibility(settings,old)
    contract,e0,backend,service,exclusive,collectors,plan,_ = node.setup_baseline(root,settings,registration)
    asset_pins = dict(service.spec.model_files)
    node.need(task['tokenizer_sha256']==asset_pins['tokenizer.json'] and
              task['template_sha256']==asset_pins['chat_template.jinja'], 'prepared workload uses another model')
    native,receipt = replay_e0.restore_build(e0_build,e0_pin,contract=contract,e0=e0,backend=backend,
                                             registration=old,registration_pin=prior_pin)
    deployment = stage_deployment(native,root/'deployment')
    node.need(deployment.identity==receipt['data']['deployment'],'screen must use the pinned E0 deployment')
    timing = collectors['develop_measure']
    required = 2*plan['warm_waves']+plan['timed_waves']
    for split, source in task['pools'].items():
        folder = root/split
        folder.mkdir(mode=0o700)
        if split == 'generalization':
            selected = {scene:(waves if scene=='W3' else waves[:required]) for scene,waves in source.items()}
            screening = None
        else:
            deadline = monotonic()+settings['node_ms']['develop_build']/1000
            with exclusive(deadline),service.start(deployment,'ee-l3-job-chat-screen-'+token_hex(8),deadline) as live:
                selected,witnesses = screen_split(source,folder,timing,live,deadline,plan,required)
            proof = write(folder/'screen.json',dict(input_manifest_sha256=inputs_pin, registration_sha256=registration,
                                                   engine=e0.digest, e0_build_sha256=e0_pin, witnesses=witnesses))
            screening = dict(engine=e0.digest,receipt_sha256=proof)
        doc = dict(schema=SCHEMA,split=split,enable_thinking=True,tokenizer_sha256=task['tokenizer_sha256'],
                   template_sha256=task['template_sha256'],screening=screening,requests=selected)
        file = folder/'requests.json'
        plan_sha = write(file,doc)
        load_chat(dict(path=str(file),sha256=plan_sha),task['tokenizer_sha256'],task['template_sha256'],
                  plan['scenes'],plan['warm_waves'],plan['timed_waves'],split=split,engine=e0.digest)
        print(split,plan_sha,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report_dir',type=Path)
    ap.add_argument('--prepared',action='store_true',help='trusted controller rendered and pinned the task bytes')
    for name in ('registration','inputs','e0-build','prior-registration'):
        ap.add_argument('--'+name,type=Path,required=True)
        ap.add_argument('--'+name+'-sha256',required=True)
    args = ap.parse_args()
    os.umask(0o077)
    cfg = node.checked_inputs(args.registration,args.registration_sha256)
    freeze(node.fresh(args.report_dir),cfg,args.registration_sha256,args.inputs,args.inputs_sha256,
           args.e0_build,args.e0_build_sha256,args.prior_registration,args.prior_registration_sha256,
           prepared=args.prepared)


if __name__=='__main__':
    main()
