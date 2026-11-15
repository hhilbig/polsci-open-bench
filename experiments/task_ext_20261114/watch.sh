#!/bin/bash
# Keep one jobstatus entry (menu bar) current for a batch of Hive Slurm jobs.
# Usage: watch.sh STATUS_ID "TASK TEXT" JOBS_FILE [LOG_PATH]
#   JOBS_FILE: one "jobid label" per line; lines can be appended while it runs.
# Polls sacct every 10 min with capped ssh retries; ends when every listed job
# is terminal. Writes --state failed from a trap if the watcher itself dies.
set -u
ID=$1; TASK=$2; JOBS=$3; LOG=${4:-}
H=hhilbig@hive.hpc.ucdavis.edu
SACCT=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin/sacct
JS="$HOME/.local/bin/jobstatus"
trap '"$JS" set "$ID" --host hive --state failed --note "watcher stopped; check sacct" >/dev/null 2>&1' TERM INT
"$JS" set "$ID" --host hive --project polsci-open-bench --task "$TASK" --unit "Slurm jobs" \
    ${LOG:+--log "$LOG"} >/dev/null 2>&1
while true; do
  ids=$(awk '{print $1}' "$JOBS" | paste -sd, -)
  out=""
  for i in $(seq 10); do
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$H" "$SACCT -X -n -P -j $ids --format=JobID,State" 2>&1); rc=$?
    [[ $rc -ne 255 ]] && break
    grep -q "Permission denied" <<<"$out" && break
    sleep 60
  done
  if [[ $rc -ne 0 ]]; then sleep 600; continue; fi
  total=$(wc -l < "$JOBS" | tr -d ' ')
  done_n=0; failed=0; note=""
  while read -r jid label; do
    st=$(awk -F'|' -v j="$jid" '$1==j{print $2}' <<<"$out" | awk '{print $1}')
    case "$st" in
      COMPLETED) done_n=$((done_n+1)) ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) done_n=$((done_n+1)); failed=$((failed+1)) ;;
    esac
    note="$note$label ${st:-?}; "
  done < "$JOBS"
  "$JS" set "$ID" --host hive --total "$total" --done "$done_n" --note "${note%; }" >/dev/null 2>&1
  if [[ $done_n -ge $total ]]; then
    if [[ $failed -gt 0 ]]; then
      "$JS" set "$ID" --host hive --state failed --note "$failed of $total failed: ${note%; }" >/dev/null 2>&1
    else
      "$JS" set "$ID" --host hive --state finished >/dev/null 2>&1
    fi
    exit 0
  fi
  sleep 600
done
