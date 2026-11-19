#!/bin/bash
# Finish the API remainder in parallel (replaces api_sequential.sh on 2026-11-16,
# when 5 GPT-6.1 Sol requests stayed open for hours). Sol's batch is only
# watched; GLM-5.3 is submitted; Gemini 4 Argon's 17 unsent task batches are
# retried. The holds for GLM (~$3.5) and Gemini (< $23.4) fit the key limit.
# The $50 cost cap in code/ext_api_batch.py applies to every submit.
#
# Run from the repository root:
#   nohup experiments/task_ext_20261114/api_parallel.sh >> output/sidecar/task_ext_20261114_api/parallel.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
RUN="python3 code/ext_api_batch.py"
CAP=50
status() { jobstatus set polsci-ext-api-seq --project polsci-open-bench \
  --task "Finish the remaining API models through OpenRouter" \
  --unit models --total 3 --host this-mac \
  --log output/sidecar/task_ext_20261114_api/parallel.log "$@" >/dev/null 2>&1 || true; }

finish() {  # $1 model, $2 command (watch|submit|retry)
  local model=$1 cmd=$2
  if [ "$cmd" != watch ]; then
    $RUN "$cmd" --stage remainder --remainder-approved --models "$model" --cost-cap-usd $CAP || return 1
  fi
  $RUN watch --stage remainder --models "$model" --interim --poll-seconds 180
}

status --state running --done 0 --note "Sol finishing, GLM submitted, Gemini retried"
finish gpt-6.1-sol watch > >(sed 's/^/[sol] /') 2>&1 & p1=$!
finish glm-5.3 submit > >(sed 's/^/[glm] /') 2>&1 & p2=$!
finish gemini-4-argon retry > >(sed 's/^/[gemini] /') 2>&1 & p3=$!
fail=0; done=0
for p in $p1 $p2 $p3; do
  if wait $p; then done=$((done + 1)); status --state running --done $done --note "$done of 3 collected"
  else fail=1; fi
done
if [ $fail -eq 0 ]; then status --state finished --done 3 --note "all remainder batches collected"; echo "$(date) all done"
else status --state failed --done $done --note "a model failed (see log)"; echo "$(date) a model failed"; exit 1; fi
