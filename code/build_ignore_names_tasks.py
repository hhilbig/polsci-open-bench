#!/usr/bin/env python3
"""
Build "ignore names" copies of the three ideological-position extension tasks.

Question: if the prompt tells the model to ignore party and politician names,
does that do what masking the names does? Each copy keeps the original data
file, so the passages are as written and the benchmark's seeded 250+250 draw
selects the same 500 items as task_ext_20261002_position and the masked run.
Only the prompt changes: IGNORE_NOTE is appended to the original prompt, with
the same wording code/run_masking_openrouter.py --ignore-names used for the
commercial models.

Writes prompts/<task>_ignore.txt and tasks_ext_ignore/<task>_ignore.yaml, and
checks that the item draw is unchanged.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import task_registry  # noqa: E402
from run_masking_openrouter import IGNORE_NOTE  # noqa: E402

TASKS = ["benoit_manifesto_economic_position", "benoit_manifesto_social_position",
         "lemens_gallego_tweet_position"]
OUT_TASKS = REPO / "tasks_ext_ignore"


def main():
    OUT_TASKS.mkdir(exist_ok=True)
    for task in TASKS:
        prompt = (REPO / "prompts" / f"{task}.txt").read_text()
        (REPO / "prompts" / f"{task}_ignore.txt").write_text(prompt.rstrip() + "\n\n" + IGNORE_NOTE)

        spec = yaml.safe_load((REPO / "tasks_ext" / f"{task}.yaml").read_text())
        spec["name"] = f"{task}_ignore"
        spec["prompt_file"] = f"../prompts/{task}_ignore.txt"
        spec["order"] = int(spec["order"]) + 200
        out = OUT_TASKS / f"{task}_ignore.yaml"
        header = (f"# Copy of tasks_ext/{task}.yaml whose prompt also tells the model to ignore\n"
                  f"# party and politician names; built by code/build_ignore_names_tasks.py.\n")
        out.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

        orig = task_registry.load_task_definitions(task_manifest=REPO / "tasks_ext" / f"{task}.yaml")[0]
        new = task_registry.load_task_definitions(task_manifest=out)[0]
        a, b = orig["loader"](), new["loader"]()
        assert [x["item_id"] for x in a] == [x["item_id"] for x in b], "item draw changed"
        assert all(x["user_content"] == y["user_content"] for x, y in zip(a, b))
        print(f"{task}: {len(b)} items, same draw and passages as the original")


if __name__ == "__main__":
    main()
