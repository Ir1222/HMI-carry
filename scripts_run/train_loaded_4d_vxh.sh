#!/usr/bin/env bash
# One recoverable segment; checkpoint files/events stay outside the repository.
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 transfer|resume CHECKPOINT UPDATES SEGMENT [NUM_ENVS=8192]" >&2
  exit 2
fi
mode=$1
checkpoint=$2
updates=$3
segment=$4
num_envs=${5:-8192}
case "$mode" in transfer) load_optimizer=False ;; resume) load_optimizer=True ;; *) exit 2 ;; esac
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
source scripts_run/loaded_4d_status.sh
export PYTHONFAULTHANDLER=1
base=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh
segment_dir="$base/$segment"
mkdir -p "$segment_dir"
python - "$checkpoint" "$updates" "$segment_dir" "$mode" "$num_envs" <<'PY'
import json,sys,torch,time
from pathlib import Path
checkpoint,updates,folder,mode,num_envs=sys.argv[1:]
p=Path(folder)/'segment.json'
if p.exists(): raise SystemExit(f'Refusing to overwrite existing segment ledger: {p}; use a new segment name')
x=torch.load(checkpoint,map_location='cpu',weights_only=False)
profile=x['experiment_config']['command']['setup_terms']['motion_command']['params']['loaded_4d'].get('profile')
if mode=='transfer' and profile=='vxh': raise SystemExit('VXH continuation must restore optimizer; use resume')
if mode=='resume' and profile!='vxh': raise SystemExit('First Stage1 transfer must use transfer')
n=int(updates)
if n<=0: raise SystemExit('updates must be positive')
p.write_text(json.dumps({'status':'running','started_unix':time.time(),'checkpoint':checkpoint,'mode':mode,'num_envs':int(num_envs),
 'requested_updates':n,'completed_updates':0,'first_display_iteration':x['iter'],'last_display_iteration':x['iter']+n-1,
 'note':'PPO repeats the saved display index on resume; sum completed_updates for the budget, not index differences.'},indent=2)+'\n')
PY
# Install after creating our ledger, so a duplicate-segment rejection cannot
# overwrite the status of an earlier run.
trap 'code=$?; if (( code != 0 )); then loaded_4d_record_failure "$segment_dir/segment.json" "$code" segment || true; fi' EXIT
python src/holosoma/holosoma/train_agent.py \
  exp:g1-loaded-4d-stage2-vxh logger:disabled \
  --training.checkpoint="$checkpoint" --training.num-envs="$num_envs" \
  --algo.config.num-learning-iterations="$updates" --algo.config.save-interval=1000 \
  --algo.config.latest-save-interval=100 \
  --algo.config.load-optimizer="$load_optimizer" \
  --algo.config.reset-policy-std-on-resume=False \
  --logger.base-dir="$segment_dir"
python - "$segment_dir" <<'PY'
import json,sys,time,torch
from pathlib import Path
folder=Path(sys.argv[1]); ledger=folder/'segment.json'; d=json.loads(ledger.read_text())
models=list(folder.glob('WholeBodyTracking/*/model_%05d.pt' % d['last_display_iteration']))
if len(models)!=1: raise SystemExit(f'Expected one final checkpoint, found {models}')
payload=torch.load(models[0],map_location='cpu',weights_only=False)
for key in ('actor_model_state_dict','critic_model_state_dict','actor_obs_normalizer_state_dict','critic_obs_normalizer_state_dict'):
    if not all(torch.isfinite(v).all() for v in payload[key].values()): raise SystemExit(f'Nonfinite saved tensor in {key}')
if not (payload['actor_model_state_dict']['std']>0).all(): raise SystemExit('Nonpositive policy std')
d.update(status='complete', completed_updates=d['requested_updates'], final_checkpoint=str(models[0]), finished_unix=time.time())
ledger.write_text(json.dumps(d,indent=2)+'\n')
(folder/'checkpoint.txt').write_text(str(models[0])+'\n')
print(json.dumps(d,indent=2))
PY
