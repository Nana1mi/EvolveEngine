import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    import reference_qkv_quality_audit as quality


class TestReferenceQKVQuality(unittest.TestCase):
    def test_candidate_log_manifest_and_idle_teardown(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for name in sorted(quality.RAW_FILES):
                data = name.encode('utf-8')
                (root/name).write_bytes(data)
                rows.append(f'{hashlib.sha256(data).hexdigest()}  {name}')
            (root/'raw-evidence.sha256').write_text('\n'.join(rows), encoding='utf-8')
            compute = root/'post-compute.csv'
            compute.write_text('', encoding='utf-8')
            self.assertEqual(set(quality.verify_candidate_artifacts(root)), quality.RAW_FILES)
            compute.write_text('42, leftover', encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'teardown'):
                quality.verify_candidate_artifacts(root)
            compute.write_text('', encoding='utf-8')
            (root/'det2.stdout').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                quality.verify_candidate_artifacts(root)

    def test_requires_candidate_identity_and_nonduplicated_api_cases(self):
        marker = 'EE_REFERENCE_QKV_QUALITY det1 expected\n'
        cases = [dict(kind='quality',case=str(i),passed=True) for i in range(6)]
        summary = dict(kind='quality_summary', passed=6,total=6)
        def log(records):
            return marker + '\n'.join(json.dumps(r, sort_keys=True) for r in [*records, summary])
        self.assertEqual(quality.read_semantic_gate(log(cases),'det1','expected'), set(map(str, range(6))))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            quality.read_semantic_gate(log(cases[:5]+[cases[0]]),'det1','expected')
        with self.assertRaisesRegex(ValueError,'identity'):
            quality.read_semantic_gate(log(cases),'det1','other')
