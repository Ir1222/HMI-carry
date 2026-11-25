# Loaded-4D Stage2 VXH — 2026-11-25

Authorized budget: 15000 additional PPO updates, including continued smoke/short training. Do not use old Stage2 weights. Baseline HEAD e0c68a8; no pre-existing local changes or applicable AGENTS.md found. Existing remote matches HEAD. GPU initially free of compute jobs.

Hypothesis: explicit forward/height/stand target quotas, tighter independent vx reward and actual Gen support/carry constraints improve command response while 50% reference Track preserves the three original motions. Keep PPO/MLP/observations/actions/physics/data fixed. Old Stage2 remains a comparison.

Pre-training evidence: real Stage1 PPO restore audit passed exact network/normalizer/std comparisons; new optimizer lr=1e-4, iter=6000, std mean=.0391273. Stage1 physical calibration and frozen criteria are recorded in docs/loaded_4d_vxh. The hand metric is box-local wrist geometry and relative speed, not proof of filtered hand-box contact or friction load.

Run one segment with scripts_run/train_loaded_4d_vxh.sh; segment.json records requested/completed updates and output checkpoint. First transfer clears optimizer only once. Every continuation uses resume. Saved display iteration repeats on reload; account budget from segment updates. Outputs: /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh.

Acceptance is task_success under frozen criteria, not survival, total return, or smoke convergence. Evaluate six fixed targets, then continuous control when responsive; compare all three Track clips at identical seeds/frame 0/mean actions/noise. Failed and unexecuted segments are explicit. Stop for invalid numerics/protocol or repeated nonresponse, retain evidence, make only bounded local changes.

Early checkpoint at103 additional updates: all six Gen goals failed within~1s, no valid steady window. Identical Stage1 Gen control also failed all six; early Track completed3/3, with increased object error. Keep current configuration through the scheduled1000-update short evaluation. No threshold or reward-weight adjustment was made in response to these failures. Sequential driver is running in screen `hmi-vxh-round`; it evaluates every1500 updates thereafter, stops for severe Track loss/three no-response windows, and performs final10-trial candidate verification. This is ongoing work, not a validated controller.
