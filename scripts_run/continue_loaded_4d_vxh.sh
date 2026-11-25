#!/usr/bin/env bash
# Sequential local train/evaluate loop, bounded at 15000 updates. No GPU process killing.
set -euo pipefail
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
base=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh
criteria=docs/loaded_4d_vxh/criteria.json
# Optional explicit checkpoint and completed total allow recovery after diagnosis.
if [ "$#" -eq 2 ]; then
  checkpoint=$1
  completed=$2
else
  while [ ! -f "$base/short_1000/checkpoint.txt" ]; do
    if ! pgrep -f '^python src/holosoma/holosoma/train_agent.py .*stage2-vxh' >/dev/null; then
      echo 'Short training ended without a completed segment; inspect short_1000.log.' >&2
      exit 1
    fi
    sleep 30
  done
  checkpoint=$(cat "$base/short_1000/checkpoint.txt")
  completed=1000
fi
while true; do
  eval_dir="$base/evaluations/updates_$completed"
  mkdir -p "$eval_dir"
  printf '{"state":"evaluating","completed_updates":%s,"checkpoint":"%s"}\n' "$completed" "$checkpoint" > "$base/progress.json"
  python scripts_run/evaluate_loaded_4d.py --checkpoint "$checkpoint" --mode gen \
    --suite vxh-targets --motion-id 1 --seconds 15 --trials 2 --seed 0 --criteria "$criteria" \
    --output "$eval_dir/targets.json" > "$eval_dir/targets.log" 2>&1
  python scripts_run/evaluate_loaded_4d.py --checkpoint "$checkpoint" --mode track \
    --trials 2 --start-frames 0 --seed 0 --output "$eval_dir/track.json" > "$eval_dir/track.log" 2>&1
  python scripts_run/summarize_loaded_4d_vxh.py "$eval_dir" --updates "$completed" \
    --history "$base/evaluation_history.jsonl" > "$eval_dir/summary.log"
  response=$(python -c 'import json,sys; print(not json.load(open(sys.argv[1]))["no_response"])' "$eval_dir/summary.json")
  if [ "$response" = True ]; then
    python scripts_run/evaluate_loaded_4d.py --checkpoint "$checkpoint" --mode gen \
      --suite vxh-sequence --motion-id 1 --seconds 70 --trials 2 --seed 0 --criteria "$criteria" \
      --output "$eval_dir/sequence.json" > "$eval_dir/sequence.log" 2>&1
    python scripts_run/summarize_loaded_4d_vxh.py "$eval_dir" --updates "$completed" \
      --history "$base/evaluation_history.jsonl" > "$eval_dir/summary.log"
  fi
  decision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$eval_dir/summary.json")
  if [ "$decision" = stop_for_diagnosis ] || [ "$completed" -ge 15000 ]; then break; fi
  updates=$((15000-completed)); if [ "$updates" -gt 1500 ]; then updates=1500; fi
  next=$((completed+updates))
  printf '{"state":"training","completed_updates":%s,"segment_updates":%s,"checkpoint":"%s"}\n' "$completed" "$updates" "$checkpoint" > "$base/progress.json"
  bash scripts_run/train_loaded_4d_vxh.sh resume "$checkpoint" "$updates" "round_$next" 8192 > "$base/round_$next.log" 2>&1
  checkpoint=$(cat "$base/round_$next/checkpoint.txt")
  completed=$next
done
# Final 10-trial verification of the best screened candidate, including failures.
checkpoint=$(cat "$base/best_checkpoint.txt")
candidate_updates=$(python -c 'import json,sys; rows=[json.loads(s) for s in open(sys.argv[1])]; print(next(r["additional_updates"] for r in rows if r["checkpoint"]==sys.argv[2]))' "$base/evaluation_history.jsonl" "$checkpoint")
final_dir="$base/evaluations/final_best"
mkdir -p "$final_dir"
python scripts_run/evaluate_loaded_4d.py --checkpoint "$checkpoint" --mode gen \
  --suite vxh-targets --motion-id 1 --seconds 15 --trials 10 --seed 0 --criteria "$criteria" \
  --output "$final_dir/targets.json" > "$final_dir/targets.log" 2>&1
python scripts_run/evaluate_loaded_4d.py --checkpoint "$checkpoint" --mode track \
  --trials 10 --start-frames 0 --seed 0 --output "$final_dir/track.json" > "$final_dir/track.log" 2>&1
python scripts_run/summarize_loaded_4d_vxh.py "$final_dir" --updates "$candidate_updates" \
  --history "$base/final_verification.jsonl" > "$final_dir/summary.log"
response=$(python -c 'import json,sys; print(not json.load(open(sys.argv[1]))["no_response"])' "$final_dir/summary.json")
if [ "$response" = True ]; then
  python scripts_run/evaluate_loaded_4d.py --checkpoint "$checkpoint" --mode gen \
    --suite vxh-sequence --motion-id 1 --seconds 70 --trials 10 --seed 0 --criteria "$criteria" \
    --output "$final_dir/sequence.json" > "$final_dir/sequence.log" 2>&1
  python scripts_run/summarize_loaded_4d_vxh.py "$final_dir" --updates "$candidate_updates" \
    --history "$base/final_verification.jsonl" > "$final_dir/summary.log"
fi
printf '{"state":"finished_or_diagnosis_required","completed_updates":%s,"decision":"%s","best_screened_checkpoint":"%s","final_summary":"%s"}\n' \
  "$completed" "$decision" "$checkpoint" "$final_dir/summary.json" > "$base/progress.json"
