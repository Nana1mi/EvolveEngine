"""Build and render one Level 3 execution timeline from recorded timestamps."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path


def execution_timeline(root, result, *, started_utc_ns=None):
    spans = []
    missing = set()

    def add(lane, label, begin, end, **data):
        if type(begin) is int and type(end) is int and begin < end:
            spans.append(dict(lane=lane, label=label, start_utc_ns=begin, end_utc_ns=end, **data))

    def epoch(value):
        return round(datetime.fromisoformat(value).timestamp() * 1e9)

    for entry in sorted((root/'web-research').glob('*-settlement.json')):
        record = json.loads(entry.read_bytes())
        call = entry.name.split('-', 1)[0]
        req = json.loads((entry.parent / f'{call}-request.json').read_bytes())
        add('Research broker', record['operation'], epoch(req['utc']), epoch(record['ended_utc']),
            status=record['status'])

    for row in result['segments']:
        segment = row['segment']
        report = row['report']
        calls = {}
        if 'pi_tool_events' not in report:
            missing.add('Pi tool timing')
        for event in report.get('pi_tool_events', []):
            key = event.get('call_sha256')
            if key is None:
                continue
            if event['event'] == 'tool_execution_start':
                calls[key] = event
            elif (start := calls.pop(key, None)) is not None:
                tool = event['tool']
                lane = 'Source edits' if tool in {'edit', 'write'} else 'Source reads' if tool == 'read' else 'Pi tools'
                add(lane, tool, start['utc_ns'], event['utc_ns'], segment=segment)
        directory = root / f'model-segment-{segment:04d}'
        for entry in sorted(directory.glob('*-result.json')):
            record = json.loads(entry.read_bytes())
            usage = record.get('usage') or {}
            if not usage.get('wall_ms'):
                continue
            start, end = (epoch(record[k]) for k in ('started_utc', 'ended_utc'))
            add('Model requests', f"Request {record['attempt']}", start, end, segment=segment)
            timing = usage.get('generation_timing') or {}
            if 'first_delta_ms' in timing and 'last_delta_ms' in timing:
                first_delta = start + timing['first_delta_ms'] * 1_000_000
                add('Model wait to first delta', f"Request {record['attempt']}", start, min(first_delta,end), segment=segment)
                add('Model stream', f"Request {record['attempt']}", first_delta,
                    min(end, start + timing['last_delta_ms'] * 1_000_000), segment=segment)
    if started_utc_ns is None:
        missing.add('Worker wall clock anchor')
    else:
        for index, job in enumerate(result['jobs'], 1):
            directory = root / 'worker-operations'
            start = json.loads((directory / f'{index:05d}-reserved.json').read_bytes())
            end = json.loads((directory / f'{index:05d}-transport.json').read_bytes())
            phase_data = {key: value for key, value in job.get('data', {}).items()
                          if key.endswith('phase_ms') or key.endswith('phases_ms')}
            lane = job['operation'].removeprefix('develop_').capitalize()
            host_ms = end['elapsed_ms']-start['elapsed_ms']
            if 'node_ms' in job:
                phase_data.update(worker_node_ms=job['node_ms'],
                                  dispatch_and_transport_ms=max(0,host_ms-job['node_ms']))
            add(lane, job['engine'][:12], started_utc_ns + start['elapsed_ms'] * 1_000_000,
                started_utc_ns + end['elapsed_ms'] * 1_000_000, status=job['status'], phase_durations=phase_data)
    if not any(span.get('phase_durations') for span in spans):
        missing.add('Build phase timing')
    return dict(schema='level3-execution-timeline-v1', missing=sorted(missing),
                spans=sorted(spans, key=lambda span: span['start_utc_ns']))


def render(timeline):
    spans = timeline['spans']
    first = min((span['start_utc_ns'] for span in spans), default=0)
    last = max((span['end_utc_ns'] for span in spans), default=1)
    width = max(1, last - first)
    groups = {}
    for span in spans:
        groups.setdefault(span['lane'], []).append(span)
    summary = []
    for name, rows in groups.items():
        bars = []
        for row in rows:
            elapsed = (row['end_utc_ns'] - row['start_utc_ns']) / 1e9
            detail = f"{row['label']} · {elapsed:.3f} s"
            if row.get('phase_durations'):
                detail += ' · ' + json.dumps(row['phase_durations'])
            left = (row['start_utc_ns'] - first) / width * 100
            length = elapsed * 1e9 / width * 100
            bars.append(f'<span style="left:{left:.5f}%;width:{length:.5f}%" title="{escape(detail)}"></span>')
        summary.append(f'<div class="row"><b>{escape(name)}</b><div class="rail">{"".join(bars)}</div></div>')
    missing = ', '.join(timeline['missing'])
    return ('<!doctype html><meta charset="utf-8"><title>Level 3 execution timeline</title>'
            '<style>body{font:16px system-ui;margin:3rem;background:#f6f8fa;color:#172036}'
            'main{max-width:110rem;margin:auto}.row{display:grid;grid-template-columns:12rem 1fr;'
            'align-items:center;min-height:2.5rem}.rail{position:relative;height:1.2rem;background:#e1e8f0;'
            'border-radius:3px}.rail span{position:absolute;height:100%;min-width:1px;background:#296a7a;'
            'border-right:1px solid white;box-sizing:border-box;border-radius:2px}</style>'
            '<main><h1>Level 3 execution timeline</h1>'
            f'<p>0 to {width / 6e10:.2f} minutes · hover over a bar for details</p>'
            + ''.join(summary) + (f'<p>Unrecorded · {escape(missing)}</p>' if missing else '') + '</main>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    recorded = args.run / 'execution-timeline.json'
    if recorded.is_file():
        timeline = json.loads(recorded.read_bytes())
    else:
        result = json.loads((args.run / 'run/result.json').read_bytes())
        timeline = execution_timeline(args.run, result)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(render(timeline))


if __name__ == '__main__':
    main()
