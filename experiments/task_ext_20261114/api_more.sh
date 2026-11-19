#!/bin/bash
# Second round of API models on the 2026-11-14 task pool (approved 2026-11-17,
# about $17): pilot (40 requests), then the remainder, each model in parallel.
# The cost cap covers all API spend in output/sidecar/task_ext_20261114_api.
#
# Run from the repository root:
#   nohup experiments/task_ext_20261114/api_more.sh MODEL... >> output/sidecar/task_ext_20261114_api/more.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
RUN="python3 code/ext_api_batch.py"
CAP=45
LOG=output/sidecar/task_ext_20261114_api/more.log
status() { jobstatus set polsci-ext-api-more --project polsci-open-bench \
  --task "Run more API models on the new benchmark tasks through OpenRouter" \
  --unit models --total $# --host this-mac --log $LOG "$@" >/dev/null 2>&1 || true; }

one() {  # pilot, then remainder; a refusal for balance or key limit (exit 75) retries every 10 minutes
  local m=$1 stage flag
  for stage in pilot remainder; do
    flag=""; [ $stage = remainder ] && flag="--remainder-approved"
    for attempt in $(seq 36); do
      $RUN submit --stage $stage $flag --models "$m" --cost-cap-usd $CAP && [ -f "output/sidecar/task_ext_20261114_api/$m/$stage/state.json" ] && break
      echo "$(date) $m $stage: submit refused; waiting 10 minutes"; sleep 600
    done
    [ -f "output/sidecar/task_ext_20261114_api/$m/$stage/state.json" ] || return 1
    $RUN watch --stage $stage --models "$m" --interim --poll-seconds 180 || return 1
  done
}

models=("$@")
jobstatus set polsci-ext-api-more --project polsci-open-bench --task "Run more API models on the new benchmark tasks through OpenRouter" \
  --unit models --total ${#models[@]} --done 0 --host this-mac --log $LOG --state running --note "pilots submitted" >/dev/null 2>&1 || true
pids=()
for m in "${models[@]}"; do one "$m" > >(sed "s/^/[$m] /") 2>&1 & pids+=($!); done
ok=0; failed=0
for p in "${pids[@]}"; do if wait $p; then ok=$((ok+1)); else failed=$((failed+1)); fi
  jobstatus set polsci-ext-api-more --project polsci-open-bench --task "Run more API models on the new benchmark tasks through OpenRouter" \
    --unit models --total ${#models[@]} --done $ok --host this-mac --log $LOG --state running --note "$ok collected, $failed failed" >/dev/null 2>&1 || true
done
state=finished; [ $failed -gt 0 ] && state=failed
jobstatus set polsci-ext-api-more --project polsci-open-bench --task "Run more API models on the new benchmark tasks through OpenRouter" \
  --unit models --total ${#models[@]} --done $ok --host this-mac --log $LOG --state $state --note "$ok collected, $failed failed" >/dev/null 2>&1 || true
echo "$(date) done: $ok collected, $failed failed"
