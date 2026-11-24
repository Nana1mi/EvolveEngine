"""Extra CUDA allocator attribution for the residence diagnostic, never a timed arm."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path


def begin():
    import torch
    # Only live states and Python stacks. The installed ARM64 runtime's native
    # frame symbolizer raised RuntimeError('stoi') on the ready-state snapshot.
    torch.cuda.memory._record_memory_history(enabled='state', context='state', stacks='python', max_entries=1)


def origin(frames):
    preferred = []
    fallback = []
    for f in frames:
        filename = f['filename'].replace('\\', '/')
        desc = f"{filename.rsplit('/', 1)[-1]}:{f['line']}:{f['name']}"
        if '/sglang/srt/' in filename:
            preferred.append('sglang/srt/'+filename.split('/sglang/srt/',1)[1]+f":{f['line']}:{f['name']}")
        elif not filename.startswith('<'):
            fallback.append(desc)
    return tuple((preferred[:4] or fallback[:3]) or ['<no allocation frames>'])


def compare_blocks(snapshot, inventory):
    active, pending = {}, 0
    for segment in snapshot['segments']:
        for b in segment['blocks']:
            status = b['state']
            if status == 'active_awaiting_free':
                pending += b['size']
            elif status == 'active_allocated':
                addr = b['address']
                if addr in active or not 0 < b['requested_size'] <= b['size']:
                    raise ValueError('Invalid active block or duplicate address')
                active[addr] = b
            elif status != 'inactive':
                raise ValueError(f'Unknown allocator state {status!r}')
    bounds = sorted(active)
    observed = defaultdict(list)
    missed = []
    role_bytes = defaultdict(int)
    for name, item in inventory.items():
        addr, size = item['address'], item['bytes']
        i = bisect_right(bounds, addr)-1
        if i < 0 or addr+size > bounds[i]+active[bounds[i]]['requested_size']:
            missed.append(name)
        else:
            observed[bounds[i]].append((name,item))
            role_bytes[item['role']] += size
    groups = defaultdict(lambda: dict(blocks=0, requested_bytes=0, allocated_bytes=0, largest_requested_bytes=0))
    matched, pad, matched_residue = 0, 0, 0
    for addr, b in active.items():
        actual, requested = b['size'], b['requested_size']
        pad += actual-requested
        if addr in observed:
            items = sorted((x['address'], x['address']+x['bytes']) for _,x in observed[addr])
            if any(items[i][1] > items[i+1][0] for i in range(len(items)-1)):
                raise ValueError('Declared storage ranges alias unexpectedly')
            matched += sum(end-start for start,end in items)
            matched_residue += requested-sum(end-start for start,end in items)
            continue
        g = groups[origin(b['frames'])]
        g['blocks'] += 1
        g['allocated_bytes'] += actual
        g['requested_bytes'] += requested
        g['largest_requested_bytes'] = max(requested, g['largest_requested_bytes'])
    return dict(active_allocated_bytes=sum(b['size'] for b in active.values()), pending_free_bytes=pending,
                matched_known_storage_bytes=matched, matched_role_bytes=dict(role_bytes), padding_bytes=pad,
                unclassified_requested_bytes_in_matched_blocks=matched_residue, inventory_not_matched=missed,
                other_live=[dict(stack=key, **r) for key,r in sorted(groups.items(), key=lambda x: -x[1]['requested_bytes'])])


def capture(data):
    import torch
    inv = dict(data['inventory'])
    for k, xs in enumerate(data['model_owned_unclassified']):
        inv[f'other_model_owned_{k}'] = dict(xs[0], role='other_model_owned')
    breakdown = compare_blocks(torch.cuda.memory._snapshot(), inv)
    free,total = torch.cuda.mem_get_info()
    output = dict(label=data['label'], **breakdown,
                  reported_live_bytes=data['allocator']['live_bytes'],
                  device_total_bytes=total, device_free_bytes=free,
                  scope='Torch allocator block stacks; device-wide free/total is context only, not attribution.')
    print('EE_ALLOCATOR '+json.dumps(output, sort_keys=True), flush=True)


def audit(log):
    entries = [json.loads(line.removeprefix('EE_ALLOCATOR ')) for line in log.splitlines() if line.startswith('EE_ALLOCATOR ')]
    if [a['label'] for a in entries] != ['after_load','ready','post_requests']:
        raise ValueError('Missing allocation trace checkpoints')
    for a in entries:
        if a['inventory_not_matched'] or a['active_allocated_bytes'] != a['reported_live_bytes']:
            raise ValueError(f'{a["label"]}: incomplete allocator inventory or byte reconciliation')
        other = sum(g['requested_bytes'] for g in a['other_live'])
        reconciled = (a['matched_known_storage_bytes']+a['padding_bytes']+
                      a['unclassified_requested_bytes_in_matched_blocks']+other)
        if reconciled != a['active_allocated_bytes']:
            raise ValueError(f'{a["label"]}: incomplete block accounting')
        a['other_requested_bytes'] = other
        a['other_unattributed_stack_bytes'] = sum(g['requested_bytes'] for g in a['other_live']
                                                if g['stack'] == ['<no allocation frames>'])
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.log.read_text(encoding='utf-8')), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
