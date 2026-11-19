"""Find every item OpenRouter's Anthropic preflight moderation rejects, using Haiku.

One flagged request makes OpenRouter fail a whole batch before running it, with
no charge, and the error names only that request. Each round records the
flagged id in output/sidecar/task_ext_20261114_api/moderation_excluded.json
(which code/ext_api_batch.py leaves out of Anthropic batches) and resubmits the
rest. A resubmission refused for lack of balance waits 10 minutes. If a reply
cannot be read, OpenRouter's batch list decides whether a batch was created.

Run from the repository root:
    python3 experiments/task_ext_20261114/probe_moderation.py
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
API_ROOT = REPO / "output/sidecar/task_ext_20261114_api"
MODEL = "claude-haiku-5.5"
STATE = API_ROOT / MODEL / "remainder/state.json"
EXCL = API_ROOT / "moderation_excluded.json"
FLAG = re.compile(r"request '(x\d{5})'.*flagged for \"([^\"]+)\"")


def run(*args):
    return subprocess.run([sys.executable, "code/ext_api_batch.py", *args], cwd=REPO).returncode


def read():
    return json.loads(STATE.read_text())


def write(state):
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def remote_haiku_batches():
    key = (REPO / ".openrouter_api_key").read_text().strip()
    req = urllib.request.Request("https://openrouter.ai/api/v1/batches?limit=100",
                                 headers={"Authorization": f"Bearer {key}"})
    data = json.load(urllib.request.urlopen(req, timeout=60))
    return [b["id"] for b in data.get("data") or [] if "claude-haiku-5.5" in (b.get("model") or "")]


def resolve_submitting(state):
    """A crash while reading a refusal leaves 'submitting'; check OpenRouter."""
    unknown = [b for b in remote_haiku_batches() if b not in state["batch_ids"]]
    if unknown:
        state["batch_ids"].extend(unknown)
        state["status"] = "submitted"
        print(f"found batch(es) created during the unreadable reply: {unknown}", flush=True)
    else:
        state["status"] = "ended"
        print("no batch was created; treating the resubmission as refused", flush=True)
    write(state)


def main():
    draw = json.loads((API_ROOT / "draw.json").read_text())["rows"]
    # On entry the last batch has already been handled (its flagged id recorded), so
    # resubmit rather than treat the old failure as new.
    refused_last = True
    for rnd in range(200):
        state = read()
        if state["status"] == "submitting":
            resolve_submitting(state)
            refused_last = read()["status"] == "ended"
            state = read()
        if state["status"] == "submitted":
            run("watch", "--stage", "remainder", "--models", MODEL, "--interim", "--poll-seconds", "60")
            refused_last = False
            state = read()
        counts = state.get("request_counts") or {}
        if not refused_last and counts.get("completed", 0) > 0:
            print(f"{MODEL} batch passed preflight: {counts}", flush=True)
            return 0
        messages = [e.get("message", "") if isinstance(e, dict) else str(e) for e in state.get("error") or []]
        flagged = {m.group(1): m.group(2) for t in messages for m in [FLAG.search(t)] if m}
        excluded = json.loads(EXCL.read_text()) if EXCL.exists() else {}
        new = {c: r for c, r in flagged.items() if c not in excluded}
        for cid, reason in new.items():
            row = draw[int(cid[1:])]
            excluded[cid] = {"provider": "anthropic", "task": row["task"], "item_id": row["item"]["item_id"],
                             "reason": f"OpenRouter preflight moderation: {reason}",
                             "found": time.strftime("%Y-%m-%d")}
        EXCL.write_text(json.dumps(excluded, indent=1) + "\n")
        print(f"round {rnd}: {state['status']} {counts}; {len(excluded)} excluded", flush=True)
        if not refused_last and not new:
            print("batch failed without a new flagged id; stopping for a manual look", flush=True)
            return 1
        rc = run("retry", "--stage", "remainder", "--remainder-approved", "--models", MODEL, "--cost-cap-usd", "50")
        refused_last = rc != 0
        if read()["status"] == "submitting":
            continue  # resolved at the top of the next round
        if refused_last:
            print(f"retry refused (exit {rc}); waiting 10 minutes", flush=True)
            time.sleep(600)
    return 1


if __name__ == "__main__":
    sys.exit(main())
