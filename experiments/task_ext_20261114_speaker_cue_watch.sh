#!/bin/bash
# Watch the six speaker-cue jobs on Hive from the primary Mac, report progress with
# jobstatus, and rsync the outputs when no job is pending or running.
# Usage (from the repository root): caffeinate -i bash experiments/task_ext_20261114_speaker_cue_watch.sh
H=hhilbig@hive.hpc.ucdavis.edu
CP=$HOME/.ssh/cm-hive-spkcue
JOBS=25735656,25735657,25735658,25735659,25735660,25735661
REMOTE=/nfs/hive/scratch/hhilbig/polsci-speakercue-20261114/output/sidecar/task_ext_20261114_speaker_cue/
LOCAL=output/sidecar/task_ext_20261114_speaker_cue/
LOG=${LOCAL}watch.log
ID=speaker-cue-20261114
SSH=(ssh -o ControlPath=$CP -o BatchMode=yes -o ConnectTimeout=20)
SACCT=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin/sacct
finish() { jobstatus set $ID --host hive --state "$1" --note "$2"; echo "$(date -Is) $1: $2" >> "$LOG"; }
trap 'finish failed "watcher stopped before the jobs ended"' TERM INT
jobstatus set $ID --host hive --project "LLM differential error" \
  --task "Label Greek election speech paragraphs with and without the speaker's name" \
  --total 6 --unit models --log "$PWD/$LOG" --note "six Hive jobs queued" --done 0
for i in $(seq 150); do
  out=$("${SSH[@]}" $H "$SACCT -j $JOBS -n -X -P -o JobID,State" 2>&1); rc=$?
  if [[ $rc -ne 0 ]]; then echo "$(date -Is) ssh rc=$rc ${out:0:120}" >> "$LOG"; sleep 240; continue; fi
  done_n=$(grep -c "COMPLETED" <<<"$out" || true)
  active=$(grep -c "PENDING\|RUNNING\|REQUEUED\|COMPLETING" <<<"$out" || true)
  echo "$(date -Is) $(tr '\n' ' ' <<<"$out")" >> "$LOG"
  jobstatus set $ID --host hive --done "$done_n" --note "$done_n of 6 models finished, $active queued or running"
  if [[ $active -eq 0 ]]; then
    for k in $(seq 10); do
      rsync -a --partial -e "ssh -o ControlPath=$CP -o BatchMode=yes -o ConnectTimeout=20" $H:$REMOTE $LOCAL >> "$LOG" 2>&1 && break
      sleep 60
    done
    if [[ $done_n -eq 6 ]]; then finish finished "6 of 6 models finished; outputs copied"
    else finish failed "$done_n of 6 models completed: $(tr '\n' ' ' <<<"$out")"; fi
    trap - TERM INT
    exit 0
  fi
  sleep 240
done
finish failed "watcher timed out after 10 hours with jobs still active"
