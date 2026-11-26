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

- CPU: 56 relevant tests passed (53 task/runner/Viser tests plus 3 Viser-before-Isaac startup regressions), including old preset isolation, observation/action compatibility, nonzero Track axes versus zero Gen axes, quota/bin coverage, shared random/manual ramps, arrival dwell, forward single support exemption, Gen masks, first failure and hidden reset, repeated trial reset, Viser regressions and task-success false-positive cases. Original data hash/fps validation passed.
- Real Viser GUI: new controls constructed, fixed zero-axis number fields accepted, temporary local HTTP returned 200. Old controls remain covered. Full rendered VXH policy quality is not claimed.
- Real simulation: 64-env transfer 3 updates, save, restore, 2 further updates. All 935 recorded smoke scalar samples finite. Adam steps 60→100 prove optimizer continuation; std mean .03954→.03955; both normalizer counts continue. The five updates count toward the 15000 budget.
- Formal short training: 8192 env, 995 additional updates started, restoring the smoke optimizer/std/normalizers. Cumulative planned short stage=1000. Capability results are pending; smoke is implementation verification, not convergence evidence.

Run root: `/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh`.
Initial smoke checkpoint: `smoke_initial/WholeBodyTracking/20261125_145316-g1_loaded_4d_stage2_vxh-locomotion/model_06002.pt`.
Restored smoke checkpoint: `smoke_resume/WholeBodyTracking/20261125_145417-g1_loaded_4d_stage2_vxh-locomotion/model_06003.pt`.
Short run: `short_1000/WholeBodyTracking/20261125_145544-g1_loaded_4d_stage2_vxh-locomotion`.
TensorBoard: http://localhost:6020, logdir is the run root. `segment.json` in each segment records updates, display labels, input and output checkpoint. The upstream PPO repeats the saved label on resume; sum completed updates rather than subtracting labels.

## Early capability measurements and ongoing work

The stable diagnostic copy `diagnostics/updates_103/checkpoint/model_06100.pt` represents **103 additional updates**, not a converged model. Six targets were measured with one trial each from locomotion frame 0, mean actions, seed 0. All failed physically with pelvis height below .40 m; falling displacement is not credited as successful movement. No target had the required >=2 s steady window, so no valid steady MAE/P95 is reported as a control score.

| Target vx / height | Success | First failure (s) | Applied reached target (s) | Pelvis start → end (m) |
|---|---:|---:|---:|---:|
| .1 / .73 | 0/1 | .98 | .92 | .741 → .374 |
| .4 / .73 | 0/1 | 1.02 | .18 | .741 → .363 |
| .8 / .73 | 0/1 | 1.02 | Not reached | .741 → .391 |
| 0 / .60 | 0/1 | .98 | Not reached | .741 → .386 |
| 0 / .69 | 0/1 | .98 | Not reached | .741 → .362 |
| 0 / .80 | 0/1 | .98 | Not reached | .741 → .346 |

The untrained Stage1 checkpoint under the identical Gen command protocol also failed all six targets in .98–1.04 s. The 103-update checkpoint completed all three Track clips (one seed each):

| Track | Complete | Body / object position MAE (m) |
|---|---:|---:|
| continuous squat | 1/1 | .05872 / .09320 |
| locomotion | 1/1 | .04295 / .04717 |
| repeated squat | 1/1 | .05996 / .09851 |

These samples do not establish final Track retention; object error is larger than the Stage1 baseline. They also do not establish a unique failure cause. The decision is to continue the planned short training rather than treat 103 updates as a convergence test. Raw Target/Applied/Measured, displacement, support, geometry, clearance and failure traces are outside Git; [early_diagnostics.json](docs/loaded_4d_vxh/early_diagnostics.json) is the small evidence summary.

At the 2026-11-25 23:12:30 CST snapshot, training PID **1140610** had completed **246/15000 additional updates**, display iteration6243; the latest saved `model.pt` held display6200 / additional203. The 8192-env short segment targets cumulative1000. Saved actor lr=1e-5 after the original adaptive schedule (initial transfer used1e-4), std mean=.04416; recorded scalars remained finite. See [training_snapshot.json](docs/loaded_4d_vxh/training_snapshot.json) for measured sampling bins and reward contributions. Early sampling was approximately49.8% forward/40.4% height/9.9% stand, with nearly uniform within-category bins. The Gen tracking constant sums to3.5/s; reward-rate logs separately expose task gains and action/contact/support/carry costs.

The sequential continuation script is already waiting in screen session **hmi-vxh-round**. After1000 updates it evaluates six goals and three Track clips with2 trials each; it continues in1500-update segments within the15000 budget, restores optimizer/std/normalizers, and stops for severe Track loss or three consecutive evaluation windows without stable command response. Candidate ranking uses task success, command errors, survival and Stage1-relative Track completion/body/object error. At budget end or a diagnostic stop, the best screened candidate receives10 trials per fixed target and Track clip; continuous10-trial evaluation requires measured stable single-target response. Outcomes and decisions are persisted under `evaluations/`, `evaluation_history.jsonl`, `final_verification.jsonl` and `progress.json` in the run root.

**Not yet completed:** cumulative1000 short-training evaluation, final10 trials per target, continuous control, and final10-trial Track retention. No VXH checkpoint is recommended as a usable controller yet. `model.pt` is a recovery artifact; the103-update copy is a failed diagnostic comparison. No higher speed, lower height, or vx+h combination is claimed.

Training/continuation/target/sequence/Track/Viser/TensorBoard commands and recovery examples are in README. First implementation commit: `c354892`; subsequent result/runner changes are recorded by Git history. Ordinary push targets `origin/feature/loaded-4d` and preserves remote history.

## 2026-11-26: 2500-update evaluation, startup crash and recovery

2500 additional updates completed (`round_2500/model_08496.pt`). Six fixed targets, two trials each:

| vx / h | Success | Mean survival (s) |
|---|---:|---:|
| .1 / .73 | 0/2 | 2.91 |
| .4 / .73 | 0/2 | 3.58 |
| .8 / .73 | 0/2 | 2.73 |
| 0 / .60 | 0/2 | 13.39 |
| 0 / .69 | 1/2 | 13.08 |
| 0 / .80 | 0/2 | 13.17 |

The .60 and .80 height commands produced mean measured heights about .698 and .712 m: longer survival is not correct height control. The .4 m/s target had vx MAE .359 m/s and still failed. Track completed2/2 for each clip; locomotion object MAE increased to .0973 m. Continuous control completed0/2 and passed0/2. The .69 result is one successful trial, not a reliable controller. [Full small summary](docs/loaded_4d_vxh/updates_2500_summary.json).

At **01:22:32 CST**, Python PID1554115 segfaulted during launch of `round_4000`, before creating a training run/config/checkpoint or completing an update. Kernel journal confirms the native crash inside python3.11. This was neither an evaluation guard stop nor budget exhaustion. The old driver exited on the child error but left `progress.json` and the segment ledger showing training/running. Original logs lacked a Python stack; the specific dependency/root cause remains unconfirmed.

On user status inspection at16:00, no training/driver process remained. The2500 checkpoint passed finite network/normalizer checks, retained both optimizers at step50000 and std mean .12408, and CUDA worked. Recovery now records failed exits and signal codes, enables Python faulthandler, reuses matching completed evaluation, preserves failed directories under new retry names, and prevents duplicate continuation drivers. Five relevant runner tests passed, including exit139 propagation and exact update accounting.

Resumed the same experiment at **16:03**, screen `hmi-vxh-recovery`, PID1617053, 8192env, segment `round_4000_retry_20261126_160321`; optimizer load and actual PPO updates8496–8498 were confirmed. The target remains cumulative4000, total budget15000. No reward/threshold/physics/policy changes were made. Native-crash recurrence is not ruled out by this successful restart; new failures will have accurate status and additional diagnostics. [Crash/recovery record](docs/loaded_4d_vxh/recovery_20261126.json).
