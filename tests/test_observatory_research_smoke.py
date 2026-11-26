import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.spark26_gguf.level3 import smoke_web_research as smoke


class FakeResponse(io.BytesIO):
    status = 200


class Provider:
    base_url = 'https://api.deepseek.com'

    def __init__(self, results, data):
        self.results, self.data, self.calls = results, data, []

    def request(self, route, *, payload, timeout):
        self.calls.append(route)
        return FakeResponse(json.dumps(self.results if route.endswith('search') else self.data).encode())


class SmokeTest(unittest.TestCase):
    def test_pinned_contents_and_request_count(self):
        code = b'// CUDA\nvoid mul_mat_vec();\n'
        p = Provider({'results': [{'url': smoke.RAW}]}, {'content': code.decode()})
        with tempfile.TemporaryDirectory() as tmp, patch.object(smoke, 'urlopen', return_value=io.BytesIO(code)):
            d = Path(tmp)/'receipt'
            s = smoke.smoke(d, p)
            self.assertTrue(s['fetch_exact_bytes'])
            self.assertTrue(s['fetch_contains_full_source'])
            self.assertEqual(p.calls, ['/v1/tools/search', '/v1/tools/fetch'])
            self.assertTrue((d/'calls/000002-settlement.json').exists())

    def test_no_retry_and_no_fetch_if_search_empty(self):
        p = Provider({'results': []}, {'content': 'not used'})
        with tempfile.TemporaryDirectory() as t, patch.object(smoke, 'urlopen', return_value=io.BytesIO(b'mul_mat_vec')):
            d = Path(t)/'receipt'
            with self.assertRaisesRegex(ValueError, 'stopping without retries'):
                smoke.smoke(d, p)
            self.assertEqual(p.calls, ['/v1/tools/search'])
            self.assertFalse(json.loads((d/'summary.json').read_bytes())['search_nonempty'])
