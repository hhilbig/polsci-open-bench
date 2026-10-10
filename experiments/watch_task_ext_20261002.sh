#!/bin/bash
# Slack watcher for the task_ext_20261002_position run on Hive (four models).
#
# Runs on the primary Mac (mac2 cannot SSH to Hive). Polls sacct every 5
# minutes and posts through the job-alerts helper on mac2, which holds the
# webhook. Posts: once at start, every 6 hours while jobs wait, once when the
# first job starts running, and once when every job has reached a terminal
# state. Jobs run on the preemptible low partition; a preempted job is requeued
# and counts as waiting, not failed. Stays silent while Hive is unreachable and
# posts one ACTION NEEDED after 30 minutes of that. Start with:
#   JOB_IDS="1,2,3,4" nohup caffeinate -i bash experiments/watch_task_ext_20261002.sh >> <log> 2>&1 &
# Set SLACK_DRY_RUN=1 to print messages instead of posting.

set -u

JOB_IDS="${JOB_IDS:?set JOB_IDS to a comma-separated list of Slurm job ids}"
HIVE="hhilbig@hive.hpc.ucdavis.edu"
W="/nfs/hive/scratch/hhilbig/polsci-taskext-20261002"
RUN_DIR="${RUN_DIR:-$W/output/sidecar/task_ext_20261002}"
PROJECT="Polsci LLM benchmark"
TASK="${TASK:-Test run of four new left-right position tasks on four open-weight models}"
POLL_SECONDS=300
PROGRESS_SECONDS=$((6 * 3600))
UNREACHABLE_SECONDS=$((30 * 60))
N_JOBS=$(echo "$JOB_IDS" | tr ',' '\n' | grep -c .)

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

query() {
    # One line per job: JobName|State|Elapsed. `|| true` keeps an empty result
    # from looking like an SSH failure.
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$HIVE" \
        "bash -lc 'sacct -j $JOB_IDS -n -X -P -o JobName,State,Elapsed || true'" 2>/dev/null
}

last_post=0
started_posted=0
unreachable_since=0
unreachable_flag=0
first=1

while true; do
    now=$(date +%s)
    lines="$(query)"
    if [ -z "$lines" ]; then
        [ "$unreachable_since" -eq 0 ] && unreachable_since=$now
        if [ $((now - unreachable_since)) -ge "$UNREACHABLE_SECONDS" ] && [ "$unreachable_flag" -eq 0 ]; then
            post "ACTION NEEDED" "Watcher cannot reach Hive for 30 minutes. The jobs themselves may be fine." \
                "Unknown until Hive is reachable." "check ssh to hive from the primary Mac"
            unreachable_flag=1
        fi
        sleep "$POLL_SECONDS"; continue
    fi
    unreachable_since=0; unreachable_flag=0

    n_done=0; n_fail=0; n_run=0; n_wait=0; failed=""
    while IFS='|' read -r name state elapsed; do
        state="${state%% *}"
        case "$state" in
            COMPLETED) n_done=$((n_done + 1)) ;;
            RUNNING|COMPLETING) n_run=$((n_run + 1)) ;;
            PENDING|REQUEUED|PREEMPTED|SUSPENDED|RESIZING) n_wait=$((n_wait + 1)) ;;
            *) n_fail=$((n_fail + 1)); failed="$failed $name ($state)" ;;
        esac
    done <<< "$lines"
    echo "$(date -u +%FT%TZ) done=$n_done running=$n_run waiting=$n_wait failed=$n_fail"

    if [ $((n_done + n_fail)) -ge "$N_JOBS" ]; then
        if [ "$n_fail" -eq 0 ]; then
            post FYI "All $N_JOBS models finished." \
                "Done. Next is copying the output back and comparing the models." \
                "partition low, output in $RUN_DIR"
            exit 0
        fi
        post "ACTION NEEDED" "$n_done of $N_JOBS models finished; these failed:$failed." \
            "Unknown until the failures are fixed." "partition low, check logs in $W/logs"
        exit 1
    fi

    if [ "$first" -eq 1 ]; then
        post FYI "Watching $N_JOBS jobs on the low partition: $n_run running, $n_wait waiting." \
            "Each model takes minutes once it gets a GPU; the queue wait is unknown." \
            "partition low, jobs $JOB_IDS"
        last_post=$now; first=0
        [ "$n_run" -gt 0 ] && started_posted=1
    elif [ "$started_posted" -eq 0 ] && [ $((n_run + n_done)) -gt 0 ]; then
        post FYI "First job has started. $n_done finished, $n_run running, $n_wait waiting." \
            "The rest follow as GPUs free up." "partition low, jobs $JOB_IDS"
        last_post=$now; started_posted=1
    elif [ $((now - last_post)) -ge "$PROGRESS_SECONDS" ]; then
        post FYI "$n_done of $N_JOBS models finished, $n_run running, $n_wait waiting in the queue." \
            "Unknown while jobs wait for a GPU." "partition low, jobs $JOB_IDS"
        last_post=$now
    fi
    sleep "$POLL_SECONDS"
done
