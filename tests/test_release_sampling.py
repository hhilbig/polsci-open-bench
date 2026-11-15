import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import release_sampling as rs  # noqa: E402


def _items(labels, key="y"):
    return [{"item_id": str(i), "user_content": "x" * (i % 37 + 1), "gt": {key: lab}}
            for i, lab in enumerate(labels)]


class StratifiedSelectTests(unittest.TestCase):
    def test_rare_class_gets_floor(self):
        items = _items(["a"] * 950 + ["b"] * 45 + ["c"] * 5)
        sel, k, counts = rs.stratified_select(items, "t", label_kind="categorical",
                                              labels=["a", "b", "c"], label_key="y", seed=1)
        self.assertEqual(len(sel), 100)
        self.assertEqual(k, 5)
        self.assertEqual(counts["c"], 5)
        self.assertGreaterEqual(counts["b"], 5)

    def test_class_smaller_than_k_takes_all(self):
        items = _items(["a"] * 498 + ["b"] * 2)
        _, _, counts = rs.stratified_select(items, "t", label_kind="categorical",
                                            labels=["a", "b"], label_key="y", seed=1)
        self.assertEqual(counts["b"], 2)

    def test_k_lowered_when_reserve_exceeds_share(self):
        labels = [f"c{i}" for i in range(20)]
        items = _items([labels[i % 20] for i in range(1000)])
        _, k, counts = rs.stratified_select(items, "t", label_kind="categorical",
                                            labels=labels, label_key="y", seed=1)
        self.assertEqual(k, 3)  # 20 * 3 = 60 <= 60; 20 * 4 = 80 > 60
        self.assertTrue(all(v >= 3 for v in counts.values()))

    def test_deterministic_and_hash_ordered(self):
        items = _items(["a"] * 300 + ["b"] * 10)
        a = rs.stratified_select(items, "t", label_kind="categorical", labels=["a", "b"],
                                 label_key="y", seed=7)[0]
        b = rs.stratified_select(list(reversed(items)), "t", label_kind="categorical",
                                 labels=["a", "b"], label_key="y", seed=7)[0]
        self.assertEqual([x["item_id"] for x in a], [x["item_id"] for x in b])
        ranks = {x["item_id"]: i for i, x in enumerate(rs._rank(items, 7, "t"))}
        order = [ranks[x["item_id"]] for x in a]
        self.assertEqual(order, sorted(order))

    def test_plain_hash_draw_when_no_rare_class(self):
        items = _items(["a", "b"] * 250)
        sel = rs.stratified_select(items, "t", label_kind="categorical", labels=["a", "b"],
                                   label_key="y", seed=3)[0]
        self.assertEqual([x["item_id"] for x in sel],
                         [x["item_id"] for x in rs._rank(items, 3, "t")[:100]])

    def test_binary_and_multi_binary(self):
        items = _items([0] * 490 + [1] * 10)
        _, _, counts = rs.stratified_select(items, "t", label_kind="binary",
                                            label_key="y", seed=1)
        self.assertEqual(counts[1], 5)
        multi = [{"item_id": str(i), "user_content": "x",
                  "gt": {"A": int(i % 2 == 0), "B": int(i % 100 == 1)}} for i in range(500)]
        _, _, counts = rs.stratified_select(multi, "t", label_kind="multi_binary",
                                            labels=["A", "B"], seed=1)
        self.assertEqual(counts["B"], 5)

    def test_pilots_are_in_sample(self):
        items = _items(["a"] * 450 + ["b"] * 50)
        sel, pilots, _, _ = rs.select_with_pilots(items, "t", label_kind="categorical",
                                                  labels=["a", "b"], label_key="y", seed=1)
        self.assertEqual(len(pilots), 2)
        self.assertTrue(pilots <= {x["item_id"] for x in sel})


if __name__ == "__main__":
    unittest.main()
