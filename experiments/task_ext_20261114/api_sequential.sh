#!/bin/bash
# Submit the refused API remainder batches one model at a time, so OpenRouter
# holds at most one model's worst-case estimate at once (largest: Opus, ~$55).
# Each model is submitted only after the previous one is collected. A refusal
# for lack of balance or key limit (exit 75) is not billed; the script waits
# 10 minutes and tries again, up to 144 times (24 hours) per model.
# The $50 cost cap in code/ext_api_batch.py applies to every submit.
#
# Run from the repository root:
#   nohup experiments/task_ext_20261114/api_sequential.sh >> output/sidecar/task_ext_20261114_api/sequential.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
RUN="python3 code/ext_api_batch.py"
CAP=50
status() { jobstatus set polsci-ext-api-seq --project polsci-open-bench \
  --task "Submit the remaining API models one at a time through OpenRouter" \
  --unit models --total 5 --host this-mac \
  --log output/sidecar/task_ext_20261114_api/sequential.log "$@" >/dev/null 2>&1 || true; }
trap 'status --state failed --note "sequential driver stopped (see log)"' ERR

step() {  # $1 model, $2 command (submit|retry), $3 models done so far
  local model=$1 cmd=$2 done=$3 rc
  for attempt in $(seq 144); do
    status --state running --done "$done" --note "$model: $cmd attempt $attempt"
    $RUN "$cmd" --stage remainder --remainder-approved --models "$model" --cost-cap-usd $CAP
    rc=$?
    # A key-limit refusal (HTTP 403) exits 0 but leaves no state file: treat it as no balance.
    if [ $rc -eq 0 ] && [ "$cmd" = submit ] && [ ! -f "output/sidecar/task_ext_20261114_api/$model/remainder/state.json" ]; then rc=75; fi
    if [ $rc -eq 0 ]; then break; fi
    if [ $rc -ne 75 ]; then echo "$(date) $model: $cmd failed with exit $rc"; status --state failed --note "$model: $cmd exit $rc"; exit $rc; fi
    echo "$(date) $model: no balance or key limit; waiting 10 minutes"
    sleep 600
  done
  [ $rc -eq 0 ] || { status --state failed --note "$model: gave up after 24 hours"; exit 1; }
  $RUN watch --stage remainder --models "$model" --interim --poll-seconds 180 || { status --state failed --note "$model: watch failed"; exit 1; }
}

# Anthropic batches fail whole if one item trips OpenRouter's moderation, so first
# find every flagged item with cheap Haiku batches (rejected batches are not billed).
status --state running --done 0 --note "finding moderation-flagged items with Haiku"
python3 experiments/task_ext_20261114/probe_moderation.py || { status --state failed --note "moderation probe stopped (see log)"; exit 1; }

i=0
for model in claude-opus-5.5 claude-sonnet-5.5 gpt-6.1-sol glm-5.3; do
  echo "$(date) starting $model"
  step "$model" submit $i
  i=$((i + 1))
done
# Gemini 4 Argon: 30 of its 47 task batches already ran; retry sends the other 17.
until $RUN watch --stage remainder --models gemini-4-argon --interim --poll-seconds 180; do sleep 300; done
echo "$(date) starting gemini-4-argon retry"
step gemini-4-argon retry $i
status --state finished --done 5 --note "all remainder batches collected"
echo "$(date) all done"
