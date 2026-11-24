import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    import allocator_residency_trace as trace


class AllocatorAttributionTests(unittest.TestCase):
    def test_counts_active_blocks_and_does_not_call_padding_a_weight(self):
        frame = {'filename':'/patched/python/sglang/srt/mem_cache/memory_pool.py', 'line':99, 'name':'initialize'}
        snapshot = dict(segments=[dict(blocks=[
            dict(address=100, size=32, requested_size=24, state='active_allocated', frames=[]),
            dict(address=132, size=8, requested_size=0, state='inactive'),
            dict(address=140, size=12, requested_size=11, state='active_awaiting_free'),
            dict(address=152, size=32, requested_size=26, state='active_allocated', frames=[frame])])])
        inventory = {'projection':dict(role='Q8_0', address=100, bytes=24)}
        report = trace.compare_blocks(snapshot, inventory)
        self.assertEqual(report['active_allocated_bytes'],64)
        self.assertEqual(report['pending_free_bytes'],12)
        self.assertEqual(report['matched_role_bytes'],{'Q8_0':24})
        self.assertEqual(report['padding_bytes'],14)
        self.assertEqual(report['other_live'][0]['requested_bytes'],26)
        self.assertEqual(report['other_live'][0]['stack'], ('sglang/srt/mem_cache/memory_pool.py:99:initialize',))
        self.assertFalse(report['inventory_not_matched'])
        rows = '\n'.join('EE_ALLOCATOR '+json.dumps(dict(label=x, reported_live_bytes=64, **report))
                         for x in ('after_load','ready','post_requests'))
        self.assertEqual(trace.audit(rows)[0]['other_unattributed_stack_bytes'],0)

    def test_padding_and_inactive_blocks_do_not_match_addresses(self):
        inv = {'bad_pad':dict(role='Q8_0', address=121, bytes=2),
               'bad_inactive':dict(role='Q8_0', address=133, bytes=2)}
        report = trace.compare_blocks(dict(segments=[dict(blocks=[
            dict(address=100,size=32,requested_size=20,state='active_allocated',frames=[]),
            dict(address=132,size=32,state='inactive')])]),inv)
        self.assertEqual(report['inventory_not_matched'],['bad_pad','bad_inactive'])
