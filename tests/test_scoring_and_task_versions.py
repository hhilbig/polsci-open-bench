"""Tests for the shared scoring module and the v2 task-manifest mechanisms.

Covers three things introduced by the 2026-09-18 audit fixes:

- `code/scoring.py`, which replaced ~10 copies of the macro-F1 computation and
  added `support_only` so labels with no gold support stop counting as zeros.
- `ground_truth.exclude_labels`, which drops gold classes that record coder
  uncertainty rather than a property of the text (GTD `Unknown`).
- `ground_truth.label_map`, which renames gold values at load time so a label
  name can be corrected without regenerating a cleaned CSV from a source archive.

It also pins the invariant that `tasks_v2/` stays isolated from `tasks/`, because
`load_task_definitions` globs a directory and a v2 file dropped into `tasks/`
would silently become a 35th task.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import scoring  # noqa: E402
import task_registry  # noqa: E402


CATEGORICAL_TASK = {
    "label_kind": "categorical",
    "label_key": "topic",
    # Four declared labels, but only two will occur in the gold column below.
    "labels": ["Health", "Defense", "Housing", "Culture"],
}

MULTI_BINARY_TASK = {
    "label_kind": "multi_binary",
    "label_key": None,
    "labels": ["Budget", "Regulatory", "Other"],
}

BINARY_TASK = {"label_kind": "binary", "label_key": "relevant", "labels": ["relevant"]}


def categorical_frame():
    # Perfect predictions on the two labels that occur.
    return pd.DataFrame(
        {
            "gt_topic": ["Health", "Health", "Defense", "Defense"],
            "pred_topic": ["Health", "Health", "Defense", "Defense"],
        }
    )


class ScoredLabelsTests(unittest.TestCase):
    def test_categorical_support_only_drops_absent_labels(self):
        got = scoring.scored_labels(CATEGORICAL_TASK, categorical_frame())
        self.assertEqual(got, ["Health", "Defense"])

    def test_categorical_legacy_keeps_every_manifest_label(self):
        got = scoring.scored_labels(
            CATEGORICAL_TASK, categorical_frame(), support_only=False
        )
        self.assertEqual(got, CATEGORICAL_TASK["labels"])

    def test_multi_binary_requires_a_positive_gold_value(self):
        frame = pd.DataFrame(
            {
                "gt_Budget": [1, 0], "pred_Budget": [1, 0],
                "gt_Regulatory": [0, 1], "pred_Regulatory": [0, 1],
                "gt_Other": [0, 0], "pred_Other": [0, 0],  # never positive
            }
        )
        self.assertEqual(
            scoring.scored_labels(MULTI_BINARY_TASK, frame), ["Budget", "Regulatory"]
        )

    def test_binary_is_unaffected(self):
        frame = pd.DataFrame({"gt_relevant": [1, 0], "pred_relevant": [1, 0]})
        self.assertEqual(scoring.scored_labels(BINARY_TASK, frame), ["relevant"])


class HeadlineF1Tests(unittest.TestCase):
    def test_zero_support_labels_drag_the_legacy_average(self):
        """The defect this module exists to fix.

        Predictions are perfect, so the metric should be 1.0. Under the legacy
        rule the two declared-but-absent labels each score 0 and pull the mean
        down to 0.5.
        """
        frame = categorical_frame()
        self.assertAlmostEqual(scoring.headline_f1(CATEGORICAL_TASK, frame), 1.0)
        self.assertAlmostEqual(
            scoring.headline_f1(CATEGORICAL_TASK, frame, support_only=False), 0.5
        )

    def test_unsupported_labels_are_nan_not_zero(self):
        per_class = scoring.per_class_f1(CATEGORICAL_TASK, categorical_frame())
        self.assertAlmostEqual(per_class["Health"], 1.0)
        self.assertTrue(np.isnan(per_class["Housing"]))
        self.assertTrue(np.isnan(per_class["Culture"]))

    def test_label_subset_pins_the_scored_classes(self):
        """A pinned subset must win over what the frame happens to contain."""
        frame = categorical_frame()
        got = scoring.headline_f1(
            CATEGORICAL_TASK, frame, label_subset=["Health", "Defense", "Housing"]
        )
        self.assertAlmostEqual(got, 2 / 3)

    def test_empty_frame_is_nan(self):
        empty = categorical_frame().iloc[0:0]
        self.assertTrue(np.isnan(scoring.headline_f1(CATEGORICAL_TASK, empty)))

    def test_macro_f1_arrays_matches_frame_scoring(self):
        frame = categorical_frame()
        labels = scoring.scored_labels(CATEGORICAL_TASK, frame)
        self.assertAlmostEqual(
            scoring.macro_f1_arrays(
                frame["gt_topic"].values, frame["pred_topic"].values, labels
            ),
            scoring.headline_f1(CATEGORICAL_TASK, frame),
        )


class MaskUnparsedTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "gt_topic": ["Health", "Defense", "Health"],
                "pred_topic": ["Health", "Defense", None],
                "parse_error": [None, None, "parse_fail: 'Terrorism'"],
            }
        )

    def test_default_drops_unparsed_rows(self):
        got = scoring.mask_unparsed(self.frame())
        self.assertEqual(len(got), 2)

    def test_unparsed_as_wrong_retains_and_sentinels(self):
        got = scoring.mask_unparsed(
            self.frame(), unparsed_as_wrong=True, pred_columns=["pred_topic"]
        )
        self.assertEqual(len(got), 3)
        self.assertEqual(got["pred_topic"].iloc[2], "__UNPARSED__")
        # The sentinel can never match a gold label, so the row scores as wrong.
        self.assertNotIn("__UNPARSED__", set(got["gt_topic"]))


class ExcludeLabelsTests(unittest.TestCase):
    def test_gtd_v2_drops_unknown_from_labels_schema_and_sample(self):
        v2 = {
            t["name"]: t
            for t in task_registry.load_task_definitions(tasks_dir=REPO / "tasks_v2")
        }
        task = v2["brandt_gtd_attack_type"]
        self.assertNotIn("Unknown", task["labels"])
        self.assertNotIn(
            "Unknown", task["json_schema"]["properties"]["attack_type"]["enum"]
        )
        gold = {item["gt"]["attack_type"] for item in task["loader"]()}
        self.assertNotIn("Unknown", gold)

    def test_v1_still_contains_unknown(self):
        v1 = {t["name"]: t for t in task_registry.load_task_definitions()}
        task = v1["brandt_gtd_attack_type"]
        self.assertIn("Unknown", task["labels"])
        gold = [item["gt"]["attack_type"] for item in task["loader"]()]
        self.assertIn("Unknown", gold)

    def test_exclude_labels_rejects_a_label_not_in_the_list(self):
        spec = {
            "label_kind": "categorical",
            "label_key": "topic",
            "labels": ["Health"],
            "text": {"template": "{text}"},
            "ground_truth": {"column": "gt", "exclude_labels": ["Nonexistent"]},
        }
        with self.assertRaisesRegex(ValueError, "not in the task's label list"):
            self._load(spec)

    def test_exclude_labels_rejected_for_binary(self):
        spec = {
            "label_kind": "binary",
            "label_key": "relevant",
            "text": {"template": "{text}"},
            "ground_truth": {"column": "gt", "exclude_labels": ["1"]},
        }
        with self.assertRaisesRegex(ValueError, "only supported for categorical"):
            self._load(spec)

    def _load(self, spec):
        import tempfile

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(yaml.safe_dump(spec))
            return task_registry.load_task_definition(path)


class LabelMapTests(unittest.TestCase):
    def test_cap_v2_renames_topic_five_and_keeps_immigration_separate(self):
        v2 = {
            t["name"]: t
            for t in task_registry.load_task_definitions(tasks_dir=REPO / "tasks_v2")
        }
        for name in ("cap_crs_policy_topic", "cap_party_platform_policy_topic"):
            task = v2[name]
            with self.subTest(task=name):
                self.assertIn("Labor", task["labels"])
                self.assertIn("Immigration", task["labels"])
                self.assertNotIn("Labor and Immigration", task["labels"])
                gold = {item["gt"]["policy_topic"] for item in task["loader"]()}
                self.assertNotIn("Labor and Immigration", gold)
                self.assertTrue(gold <= set(task["labels"]))

    def test_label_map_preserves_item_pairing_with_v1(self):
        """Renaming a label must not change which rows are sampled."""
        v1 = {t["name"]: t for t in task_registry.load_task_definitions()}
        v2 = {
            t["name"]: t
            for t in task_registry.load_task_definitions(tasks_dir=REPO / "tasks_v2")
        }
        for name in ("cap_crs_policy_topic", "cap_party_platform_policy_topic"):
            with self.subTest(task=name):
                self.assertEqual(
                    [i["item_id"] for i in v1[name]["loader"]()],
                    [i["item_id"] for i in v2[name]["loader"]()],
                )

    def test_label_map_rejects_a_target_outside_the_label_list(self):
        spec = {
            "label_kind": "categorical",
            "label_key": "topic",
            "labels": ["Health"],
            "text": {"template": "{text}"},
            "ground_truth": {"column": "gt", "label_map": {"Old": "Nonexistent"}},
        }
        with self.assertRaisesRegex(ValueError, "maps onto labels that"):
            ExcludeLabelsTests()._load(spec)


class TaskVersionIsolationTests(unittest.TestCase):
    def test_active_task_set_excludes_held_out_tasks(self):
        """33 active tasks; halterman_ccc_protest is held out (see CHANGELOG)."""
        active = task_registry.load_task_definitions(active_only=True)
        everything = task_registry.load_task_definitions()
        self.assertEqual(len(active), 33)
        self.assertEqual(len(everything), 34)
        self.assertNotIn("halterman_ccc_protest", {t["name"] for t in active})
        self.assertIn("halterman_ccc_protest", {t["name"] for t in everything})

    def test_an_excluded_manifest_is_still_loadable_when_named(self):
        """Naming it explicitly must work, or it looks like a missing file."""
        got = task_registry.load_task_definitions(
            task_manifest=REPO / "tasks" / "halterman_ccc_protest.yaml"
        )
        self.assertEqual([t["name"] for t in got], ["halterman_ccc_protest"])

    def test_v2_manifests_are_not_visible_from_the_v1_directory(self):
        v1_names = {t["name"] for t in task_registry.load_task_definitions()}
        v2_names = {
            t["name"]
            for t in task_registry.load_task_definitions(tasks_dir=REPO / "tasks_v2")
        }
        # v2 reuses v1 task names on purpose, so v1/v2 panels join on task name.
        # The isolation that matters is that loading tasks/ does not pick up
        # tasks_v2/ and end up with duplicate or extra tasks.
        self.assertTrue(v2_names <= v1_names)
        self.assertEqual(len(v1_names), 34)  # includes the held-out task

    def test_every_v2_manifest_declares_what_it_supersedes(self):
        import yaml

        for path in sorted((REPO / "tasks_v2").glob("*.yaml")):
            with self.subTest(manifest=path.name):
                spec = yaml.safe_load(path.read_text())
                self.assertEqual(spec.get("version"), 2)
                superseded = REPO / "tasks_v2" / spec["supersedes"]
                self.assertTrue(
                    superseded.resolve().exists(),
                    f"{path.name} supersedes a manifest that does not exist",
                )

    def test_v2_prompts_exist_and_record_their_provenance(self):
        for task in task_registry.load_task_definitions(tasks_dir=REPO / "tasks_v2"):
            with self.subTest(task=task["name"]):
                text = Path(task["prompt_path"]).read_text()
                self.assertTrue(Path(task["prompt_path"]).exists())
                self.assertIn("Source:", text)


if __name__ == "__main__":
    unittest.main()
