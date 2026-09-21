import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from audit_external_human_vote_tasks import majority_share, summarize_shares  # noqa: E402


class ExternalHumanVoteAuditTests(unittest.TestCase):
    def test_majority_share(self):
        self.assertEqual(majority_share(["a", "a", "b"]), 2 / 3)
        self.assertEqual(majority_share(["a", "a", "a"]), 1)

    def test_summary(self):
        result = summarize_shares("fixture", pd.Series([0.5, 0.75, 1.0, 1.0]))
        self.assertEqual(result["n_agreement_units"], 4)
        self.assertEqual(result["agreement_unit"], "items")
        self.assertEqual(result["median_majority_share"], 0.875)
        self.assertEqual(result["share_unanimous"], 0.5)


if __name__ == "__main__":
    unittest.main()
