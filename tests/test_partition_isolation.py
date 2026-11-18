from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.journal import Journal
from evolveengine.partitions import Partitions, load_bank


class PartitionTests(unittest.TestCase):
    def test_roles_and_private_salt(self):
        left = Partitions(3, b"a" * 32)
        right = Partitions(3, b"b" * 32)
        for role, pool in (("search", "T"), ("updater", "D"), ("development", "C"),
                           ("causal", "Q"), ("admission", "Q")):
            with self.assertRaises(PermissionError):
                left.block(role, pool, "a", 8)
        self.assertEqual(left.block("search", "S", "k", 8), right.block("search", "S", "k", 8))
        for pool, role in (("D", "development"), ("T", "final"), ("Q", "final")):
            rows = left.block(role, pool, "k", 8)["prompts"]
            self.assertEqual(sorted(map(len, rows)), [128, 128, 512, 512, 512, 512, 2048, 2048])
            self.assertNotEqual(rows, right.block(role, pool, "k", 8)["prompts"])

    def test_cannot_substitute_a_d_snapshot_for_c(self):
        with TemporaryDirectory() as directory, Journal(Path(directory) / "journal.sqlite") as journal:
            new_id = journal.put("bank_C", [{"lineage": "bank/D/not-independent"}])
            with self.assertRaises(ValueError):
                load_bank(journal, "causal", "C", new_id)
