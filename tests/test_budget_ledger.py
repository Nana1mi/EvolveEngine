from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.journal import InsufficientBudget, Interrupted, Journal
from evolveengine.schema import Cost, Result


class BudgetTests(unittest.TestCase):
    def test_releases_unused_and_deduplicates(self):
        with TemporaryDirectory() as directory, Journal(Path(directory) / "journal.sqlite") as journal:
            limits = (("arm", Cost(15, 10, 100, blocks=2)),)
            answer = Result({"ok": True}, Cost(5, 3, 45, 5, 1))
            cap = Cost(10, 7, 50, 40, 1)
            key = "arm/fork/call"
            journal.call(key, {"input": 1}, cap, limits, "evolving", "s2", lambda: answer)
            recovered = journal.call(key, {"input": 1}, cap, limits, "evolving", "s2",
                                     lambda: self.fail("replayed an external call"))
            self.assertEqual(answer, recovered)
            self.assertEqual(journal.spent("arm"), answer.cost)
            self.assertEqual(journal.spent("arm/fork"), answer.cost)
            with self.assertRaises(InsufficientBudget):
                journal.call("arm/other", {}, cap, limits, "evolving", "s2",
                             lambda: self.fail("started an unaffordable call"))
            self.assertTrue(journal.can_reserve(limits, Cost(10, 7, 50, blocks=1)))
            with self.assertRaises(ValueError):
                journal.call(key, {"input": 2}, cap, limits, "evolving", "s2", lambda: answer)

    def test_interruption_charges_once(self):
        def interrupt(key):
            raise Interrupted(key)

        cost = Cost(7, 3, 8, 10, 1)
        scopes = (("arm", cost),)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite"
            with self.assertRaises(Interrupted), Journal(path, interrupt) as journal:
                journal.call("arm/action", {}, cost, scopes, "fixed", "s1", lambda: Result({}, cost))
            with Journal(path) as journal:
                result = journal.call("arm/action", {}, cost, scopes, "fixed", "s1",
                                      lambda: self.fail("reran an uncertain operation"))
                self.assertEqual(result.status, "uncertain")
                self.assertEqual(journal.spent("arm"), cost)
                self.assertEqual(len(list(journal.budget_rows())), 1)
