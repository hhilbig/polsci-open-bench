#!/bin/bash
# Slack watcher for the task_ext_20261002_position pilot on Hive.
#
# Runs on the primary Mac (mac2 cannot SSH to Hive). Polls sacct every 5
# minutes and posts through the job-alerts helper on mac2, which holds the
# webhook. Posts: once at start, every 6 hours while the job waits or runs,
# once when it starts running, and once at the terminal state. Stays silent
# while Hive is unreachable and posts one ACTION NEEDED after 30 minutes of
# that. Start with:
#   nohup caffeinate -i bash experiments/watch_task_ext_20261002.sh >> <log> 2>&1 &
# Set SLACK_DRY_RUN=1 to print messages instead of posting.

set -u

JOB_ID="${JOB_ID:-24312533}"
HIVE="hhilbig@hive.hpc.ucdavis.edu"
W="/nfs/hive/scratch/hhilbig/polsci-taskext-20261002"
MODEL_DIR="$W/output/sidecar/task_ext_20261002/qwen3_30b_a3b_instruct_2507_fp8"
PROJECT="Polsci LLM benchmark"
TASK="Test run of four new left-right position tasks on open-weight models, pilot with Qwen3 30B"
POLL_SECONDS=300
PROGRESS_SECONDS=$((6 * 3600))
UNREACHABLE_SECONDS=$((30 * 60))

post() {
    # post LEVEL STATUS ETA WHERE
    # mac2's login shell is zsh, where VAR=x before `.` scopes the variable to
    # the source step only. Export inside an explicit bash instead, so the
    # dry-run flag and labels reach slack_status.
    local inner
    inner="export HOST_LABEL=hive SLACK_REPO=hhilbig/polsci-open-bench${SLACK_DRY_RUN:+ SLACK_DRY_RUN=1}; . ~/bin/slack_status.sh && slack_status $(printf '%q ' "$1" "$PROJECT" "$TASK" "$2" "$3" "$4")"
    ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "bash -c $(printf '%q' "$inner")" || echo "$(date -u +%FT%TZ) post failed"
}

hours_since() {
    # Hours, one decimal, between a Slurm timestamp and now.
    python3 - "$1" <<'PY'
import sys, datetime as dt
t = dt.datetime.fromisoformat(sys.argv[1])
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=-7)))  # Hive reports US Pacific
t = t.replace(tzinfo=now.tzinfo)
h = (now - t).total_seconds() / 3600
print(f"{h:.1f} hours" if h >= 1 else f"{round(h * 60)} minutes")
PY
}

query() {
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$HIVE" \
        "bash -lc 'sacct -j $JOB_ID -n -X -P -o State,Submit,Start,Elapsed,Partition,ExitCode'" 2>/dev/null | head -1
}

WHERE_PENDING="partition high, job $JOB_ID"
last_post=0
last_state=""
unreachable_since=0
unreachable_flag=0

while true; do
    now=$(date +%s)
    line="$(query)"
    if [ -z "$line" ]; then
        [ "$unreachable_since" -eq 0 ] && unreachable_since=$now
        if [ $((now - unreachable_since)) -ge "$UNREACHABLE_SECONDS" ] && [ "$unreachable_flag" -eq 0 ]; then
            post "ACTION NEEDED" "Watcher cannot reach Hive for 30 minutes. The job itself may be fine." \
                "Unknown until Hive is reachable." "check ssh to hive from the primary Mac"
            unreachable_flag=1
        fi
        sleep "$POLL_SECONDS"; continue
    fi
    unreachable_since=0; unreachable_flag=0
    IFS='|' read -r state submit start elapsed partition exitcode <<< "$line"
    state="${state%% *}"
    echo "$(date -u +%FT%TZ) $state"

    case "$state" in
        PENDING)
            if [ "$last_state" = "" ] || [ $((now - last_post)) -ge "$PROGRESS_SECONDS" ]; then
                post FYI "Waiting in the Hive queue for $(hours_since "$submit"). The group's GPU allowance on the high partition is fully used by other users, so the job cannot start yet." \
                    "Unknown until the job starts. The pilot then takes minutes, and the other three models under an hour." \
                    "$WHERE_PENDING"
                last_post=$now
            fi ;;
        RUNNING)
            if [ "$last_state" != "RUNNING" ]; then
                post FYI "Started after waiting $(hours_since "$submit") in the queue. Running model 1 of 4 on 5 tasks." \
                    "Within minutes. The other three models are submitted after a check of the pilot output." \
                    "partition $partition, job $JOB_ID"
                last_post=$now
            elif [ $((now - last_post)) -ge "$PROGRESS_SECONDS" ]; then
                post "ACTION NEEDED" "Still running after $elapsed elapsed, far longer than the expected few minutes." \
                    "Unknown." "partition $partition, job $JOB_ID, check logs in $W/logs"
                last_post=$now
            fi ;;
        COMPLETED)
            rows=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HIVE" \
                "cat $MODEL_DIR/tasks/*.csv 2>/dev/null | grep -c . " 2>/dev/null || echo "")
            post FYI "Finished after $elapsed of run time. Output written for the pilot model." \
                "Done. Next is a check of the pilot output, then the other three models." \
                "partition $partition, job $JOB_ID, output in $MODEL_DIR"
            echo "rows incl headers: $rows"
            exit 0 ;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
            post "ACTION NEEDED" "Ended with state $state after $elapsed. No results to use yet." \
                "Unknown until the failure is fixed." \
                "partition $partition, job $JOB_ID, check logs in $W/logs"
            exit 1 ;;
    esac
    last_state="$state"
    sleep "$POLL_SECONDS"
done
