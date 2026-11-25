# Stage2 VXH implementation and experiment record

Date: 2026-11-25. Baseline code: `e0c68a8`, branch `feature/loaded-4d`. The previous Stage2 analysis remains a historical read-only audit; this record documents the subsequent implementation and physical experiments.

## Implemented protocol

New registered entry: `exp:g1-loaded-4d-stage2-vxh`. Track/Gen=50/50. Only Gen uses forward/height/stand quotas .50/.40/.10, vx .1–.8 at h=.73, stationary h=.60–.80. Four-dimensional observations/action/normalizers/data contract, PPO/MLP/PD/200 Hz physics/50 Hz policy remain identical to Stage1. Track retains reference-derived vy/yaw-rate and original tracking rewards. No reference/asset changes, object binding, or hidden physical writes were introduced.

Random and external targets share retargetable rates [.3,.3,.6,.06] command units/s. Gen reset zeroes command vy/yaw (task/applied/target), not physical velocity. Dwell 3–5 s starts on arrival. Categories and within-category target bins are logged. Future combined targets require explicit configured rows; no height-dependent vx rewriting exists in this preset.

Independent additive task rewards: vx sigma=.10 weight=2; pelvis height .05/1; zero vy .10/.3; zero heading yaw-rate .30/.3. Gen-only support/carry penalties each weight=-.5. Existing regularization, physical failure and 3.5/s Gen tracking constant stay intact. RewardManager alone applies dt. Gen reward contributions are logged in units per second.

Support reads reordered ankle-roll net vertical force (the collisionless foot-contact-point bodies are excluded), four sole probes at ankle-local z=-.035 m, and foot-point planar velocity. Applied |vx|<.03 activates the constraint only after .5 s grace and .5 s ramp. Normal forward single support is exempt. Carry uses actual box-local wrist geometry, relative wrist/box velocity and box axis tilt; this is **geometry, not evidence of filtered hand-box contact or friction load**. Reference box local z points down; tilt uses acos(abs(z_axis dot world_up)), not the directed angle near pi.

## Frozen calibration and initialization

Initialization checkpoint:
`/home/han/pyx/HMI_clean_runs/loaded_4d/stage1/WholeBodyTracking/20261124_105123-g1_loaded_4d_stage1_joint-locomotion/model_06000.pt`.

Real PPO.load audit: Actor/Critic and both normalizers exactly equal saved tensors; learned std restored (mean .0391272567); new optimizer state empty; actor/critic lr=.0001; saved/display iteration=6000. Full simulator verifies the task contract. No old Stage2 checkpoint is used.

Acceptance criteria were frozen **before the first Stage2 update** in [criteria.json](docs/loaded_4d_vxh/criteria.json); calibration derivation and reference/rollout statistics are in [calibration_summary.json](docs/loaded_4d_vxh/calibration_summary.json). Support: each foot >=20 N, highest sole probe <=.025 m, slip <=.15 m/s for >=90% of settled stationary samples. Carry geometry bounds have a .05 m acceptance margin, relative speed <=.35 m/s, axis tilt <=.32 rad for >=95% of settled samples. Velocity MAE <=max(.03,.15*target), height MAE<=.02, actual positive movement/height response, |vy| mean<=.05, |heading yaw-rate| mean<=.15. At least 2 s of steady samples, after target arrival plus .8 s. No fall/drop/reset allowed. These thresholds will not be relaxed to pass a trained checkpoint.

Stage1 baseline, deterministic mean actions, preserved observation noise, seed 0–9, frame 0, 10 trials per reference:

| Track reference | Complete | Body position error mean (m) | Object position error mean (m) |
|---|---:|---:|---:|
| continuous_grasped_squat_stand | 9/10 | .04856 | .05726 |
| loaded_locomotion | 10/10 | .03686 | .03376 |
| repeated_grasped_squat_stand | 10/10 | .04842 | .05523 |

The failed continuous squat trial was seed 0 at 3.90 s, termination `bad_tracking`. This is a measured baseline limitation, not a Stage2 regression. Full traces remain in `stage2_vxh/calibration/stage1_track.json` outside Git.

## Validation and current training state

- CPU: 51 relevant tests passed, including old preset isolation, observation/action compatibility, nonzero Track axes versus zero Gen axes, quota/bin coverage, shared random/manual ramps, arrival dwell, forward single support exemption, Gen masks, first failure and hidden reset, repeated trial reset, Viser regressions and task-success false-positive cases. Original data hash/fps validation passed.
- Real Viser GUI: new controls constructed, fixed zero-axis number fields accepted, temporary local HTTP returned 200. Old controls remain covered. Full rendered VXH policy quality is not claimed.
- Real simulation: 64-env transfer 3 updates, save, restore, 2 further updates. All 935 recorded smoke scalar samples finite. Adam steps 60→100 prove optimizer continuation; std mean .03954→.03955; both normalizer counts continue. The five updates count toward the 15000 budget.
- Formal short training: 8192 env, 995 additional updates started, restoring the smoke optimizer/std/normalizers. Cumulative planned short stage=1000. Capability results are pending; smoke is implementation verification, not convergence evidence.

Run root: `/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh`.
Initial smoke checkpoint: `smoke_initial/WholeBodyTracking/20261125_145316-g1_loaded_4d_stage2_vxh-locomotion/model_06002.pt`.
Restored smoke checkpoint: `smoke_resume/WholeBodyTracking/20261125_145417-g1_loaded_4d_stage2_vxh-locomotion/model_06003.pt`.
Short run: `short_1000/WholeBodyTracking/20261125_145544-g1_loaded_4d_stage2_vxh-locomotion`.
TensorBoard: http://localhost:6020, logdir is the run root. `segment.json` in each segment records updates, display labels, input and output checkpoint. The upstream PPO repeats the saved label on resume; sum completed updates rather than subtracting labels.

## Capability results (not yet measured for trained VXH)

| Target vx / height | Trials | task_success | Status |
|---|---:|---:|---|
| .1 / .73 | 0 | — | Pending short-training evaluation |
| .4 / .73 | 0 | — | Pending short-training evaluation |
| .8 / .73 | 0 | — | Pending short-training evaluation |
| 0 / .60 | 0 | — | Pending short-training evaluation |
| 0 / .69 | 0 | — | Pending short-training evaluation |
| 0 / .80 | 0 | — | Pending short-training evaluation |

Continuous sequence and new checkpoint Track retention are pending. No trained VXH checkpoint is recommended for deployment yet. Keep Stage1 model_06000 as the Track reference baseline; smoke checkpoints only verify the implementation. No larger velocity, lower height or vx+h combination is claimed.

Reproducible training, continuation, target/sequence/Track evaluation and Viser/TensorBoard commands are in README. Raw checkpoints/events/traces remain outside Git; small evidence summaries and frozen configuration are under `docs/loaded_4d_vxh`.
