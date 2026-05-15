from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import task_inventory  # noqa: E402


class TaskInventoryDocTests(unittest.TestCase):
    def test_task_inventory_doc_is_current(self):
        expected = task_inventory.render_markdown(REPO)
        actual = (REPO / "docs" / "task_inventory.md").read_text()
        self.assertEqual(actual, expected)
