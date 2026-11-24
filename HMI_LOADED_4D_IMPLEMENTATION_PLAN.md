# HMI Loaded-4D Implementation Plan（审查与实施记录）

审查日期：2026-11-24（按最新请求复核更新）。状态：**最小代码实现、README训练指令及CPU验证已完成；真实物理与训练验收尚未执行**。训练指标截止本轮读取的 2026-11-24 17:23:56（Asia/Shanghai），不据日志时间推断当前进程是否仍在运行。

技术审查阶段仅读取本地源码、Git 元数据、已有 JSON/YAML/日志与 TensorBoard scalars，并使用 NumPy 检查指定三条 NPZ；未启动 simulator、训练、GPU 实验，未加载大 checkpoint tensor，未安装依赖或修改原始资产。日志可能由用户正在进行的训练继续更新，以下指标与 Git 表格保留审查时快照。

文档维护按用户后续授权执行：最初在 `HMI_clean` 的 `docs/loaded-4d-plan` 分支跟踪同名报告，当前最新代码与报告位于 `feature/loaded-4d` 分支，同时同步保留 `/home/han/pyx/HMI_LOADED_4D_IMPLEMENTATION_PLAN.md`。每次可审查的阶段性修改写入本文修改记录，并在 commit message 中说明修改内容、原因和核验结果。用户已进一步明确授权按本计划实施代码、阶段性commit并在结束后push。本轮开展代码实现与CPU验证，尚未启动仿真或训练。代码工作目录为 `/home/han/pyx/HMI_clean`，核心 Python 路径为 `/home/han/pyx/HMI_clean/src/holosoma/holosoma`；原路径报告与仓库内同名副本继续同步维护。

路径约定（下文源码引用均相对于这些已核实的绝对目录）：

- `U = /home/han/pyx/HMI_clean/src/holosoma/holosoma`
- `P = /home/han/pyx/g1_carry_baseline_new/HMI`
- `O = /home/han/pyx/g1_carry_baseline_new/HMI/src/holosoma/holosoma`
- `A = /home/han/pyx/g1_carry_baseline_new/artifacts`
- `R = /home/han/pyx/HMI_clean_runs`

标记：**事实**指本轮源码/数据/记录支持；**用户确认**指用户提供的实际运行观察；**推断**指由上述证据推导；**待验证**指必须通过未来测试或实验确认。历史报告中的 PASS 只在其原实验范围内有效。

## 1. Executive Summary

推荐从 `HMI_clean` 的 `c353731999b3578c41ad5a00f896415b45e6a9f5` 派生一个独立 Loaded-4D preset，沿用 `WholeBodyTrackingManager`、`HybridMotionCommand`、原 PPO/MLP、normalizer、固定 default-pose joint-position action、PD 和 motion loaders。**不要迁移旧 carry 环境、旧 preset 或 PPO subclass 整套实现。** 原 Object preset 必须继续可运行。

新接口只有 `g=[vx,vy,yaw_rate,pelvis_height]`；Actor 可见实际机器人/箱体状态，无 object endpoint、reference pose、motion ID、phase、intent 或 release state。Stage 1 全 Track，Stage 2 固定 Track/Gen 分组、同一 policy 和输入协议。Gen 的 command 更新时间、reference clip 时间和物理 reset 必须解耦。

最小必要改动集中在：四维 command 与严格坐标定义；观测 terms 替换；精确 reference reset 和时序/clip-end 语义；Gen command rewards 与 reference-independent physical failure；独立 preset 与评估。预计 **2 个现有源码文件的小改动、3 个任务专用新源码文件、1 个测试文件、1 个评估入口** 即可形成首版，不建立新训练框架。接触传感器增强、history、lookahead、1 ms physics、delta action 均不作为默认必做项。

原版复现状态：**用户已明确确认当前 Object Stage 1 训练结果通过 viewer 人工验证，可作为已验证的原版复现对照**。本轮没有自行运行 viewer；尚未记录该次验证的准确 checkpoint 和多起点定量指标，不据此否定用户确认，也不虚构成功率。

新任务仍有独立验证要求：三条 reference 的格式可用，但并不构成通用四维运动覆盖。行走抓持与蹲起抓持差异明显，已有失败 probe 不能证明所有控制器都不可行，P0 蹲起验证也不能证明原 HMI PD 下可学。原版 viewer 成功不自动证明这三条 loaded reference 的 joint tracking 或连续四维 Gen 已达标。

## 2. Local Repository and Reproduction Status

### 2.1 实际目录、Git 和相互关系

下表为文档纳入 Git 前的审查快照。随后按用户授权，从同一 `c353731` 基线建立 `docs/loaded-4d-plan` 文档分支；该分支仅提交本报告，原 `main` 和原算法源码保持不变。当前文档 commit 以该分支的 Git log 为准。

| 实际路径 | 内容/用途 | Commit、branch、dirty status |
|---|---|---|
| `/home/han/pyx/HMI_clean` | 上游 HMI 源码及原始 motions/assets；本次对照基线 | HEAD detached at `c353731999b3578c41ad5a00f896415b45e6a9f5`；`main` 同 commit；`git status --short` 空 |
| `/home/han/pyx/HMI_clean_runs` | 外置 launcher、preflight、smoke/capacity/formal training 结果 | 不是 Git repository；无 commit/branch |
| `/home/han/pyx/HMI_clean_runtime` | 独立运行环境 `env/`、IsaacLab、activation script | 该目录不是 Git repository；内层 `IsaacLab` 为 detached HEAD `3c6e67bb5c7ada942a6d1884ab69338f57596f77`（branch -avv 仅列该 detached HEAD），有且已核实 `source/isaaclab/setup.py` 的 flatdict 版本补丁 |
| `/home/han/pyx/g1_carry_baseline_new` | 旧工作区容器：`HMI/` 源码、`artifacts/`、`envs/g1_hmi_carry/`、`IsaacLab/`、`tools/` | 该目录不是 Git repository |
| `/home/han/pyx/g1_carry_baseline_new/HMI` | 旧修改的实际 Git repository | `1188ae57e5e5287348c1448466c8678ad827f901`，branch `feature/unified-hoi-stage1`；本轮 `status --short` 空 |
| `/home/han/pyx/g1_carry_baseline_new/IsaacLab` | 旧实验 IsaacLab | 同 `3c6e67b...`；`source/isaaclab/setup.py` dirty |

`HMI_clean` 当前 origin 的 fetch/push URL 均为 **`https://github.com/Ir1222/HMI-carry.git`**，是此前按用户明确指令完成的 remote-only 修改；本轮没有修改 Git 配置。源码仍是原上游 `https://github.com/jiashunwang/Hybrid-Motion-Imitation` 的固定 commit，工作区clean、HEAD detached。`remote -v` 当前没有单独的 upstream remote。

本地 branches/ref：`main=c353731`、`origin/main=c353731`、`origin/locomotion=f89b379`、`origin/three-point-track=5e01bfc`。这些 `origin/*` refs 是修改 URL 前遗留的本地缓存；未 fetch，**不能用它们证明新 HMI-carry 远程已经存在对应 branches/commits**。原审查阶段未 checkout、commit、push 或 fetch；随后仅按追加授权建立文档分支并提交报告，记录见§10.1。推送目标为用户指定的 HMI-carry。D001仅完成本地提交；用户随后明确授权首次推送，范围为原版代码的 `main` 与包含本报告的 `docs/loaded-4d-plan`，记录见D002。

旧仓库 origin 为 `https://github.com/Ir1222/carry_baseline.git`，另有 upstream 指向原 HMI。已有本地 branches：

| Branch | 本地 tip | 本地保存的 tracking ref |
|---|---|---|
| `feature/unified-hoi-stage1`（当前） | `1188ae5` | origin 同 tip |
| `feature/fullbody-goal-carry-autotrain` | `6c3eed6` | origin 同 tip |
| `carry/goal-conditioned-continuous` | `06f33c4` | origin 同 tip |
| `reference/continuous-side-grasp-squat-stand` | `ed106a7` | 未列 tracking ref |
| `main` | `c353731` | upstream/main 同 tip |

这只证明当前 worktree 无可见未提交改动、当前 branch 与本地 remote-tracking ref 一致，**不是对 GitHub 实时推送状态的验证**。未跟踪/被忽略的 artifacts、checkpoint 和数据直接按本地实际内容审查，未用远程内容替代。

`R/preflight.json` 记录 clean runtime 从旧 `envs/g1_hmi_carry` 克隆，但 editable bindings 已换到 clean HMI 与独立 IsaacLab。`HMI_clean_runtime/activate.sh` 清理旧 PYTHONPATH/LD_LIBRARY_PATH，绑定 clean source，并设置 `PYTHONDONTWRITEBYTECODE=1`。环境记录为 Python 3.11.15、Torch 2.7.0+cu128、Isaac Sim 5.1.0、IsaacLab v2.3.0。实际 IsaacLab diff 仅 `flatdict==4.0.1 → 4.1.0`。这是依赖适配，不是 HMI 算法修改；preflight 中记录的旧origin URL也是当时的历史快照，不覆盖本轮实际remote核验。`capacity/runtime_paths.json` 与 preflight 提供旧代码路径隔离证据。

### 2.2 原始 Object Stage 1 / Stage 2 入口

实际训练入口：`U/train_agent.py`；experiment 注册：`U/config_values/experiment.py`；preset：`U/config_values/wbt/g1/experiment.py:g1_29dof_wbt_hybrid_object`。**两阶段使用同一 preset，通过 CLI 改 Track ratio 和 checkpoint，不存在必须重写的独立 Stage 2 trainer。**

`HMI_clean/README.md` 与 `scripts_run/run_train.sh` 的 Object recipes：

- Stage 1：`exp:g1-29dof-wbt-hybrid-object`，8192 envs，15000 updates，scene spacing 4.0，`hybrid.track_ratio=1.0`，`motion_dir=holosoma/data/motions/g1_29dof/whole_body_tracking/movebox`，frame0 reset probability 0.05，size randomization disabled。
- Stage 2：同 preset，`--training.checkpoint=...`，`track_ratio=0.5`，Gen frame0 probability 0.2；`reset_policy_std_on_resume=True`、`init_noise_std=0.45`；object goal position noise std `[0.5,0.5,0.001]`、clip `[1,1,0.001]`，RPY noise std `[0.001,0.001,0.4]`、clip `[0.001,0.001,0.8]`；启用 endpoint noise curriculum。Loaded-4D 不复用这些 endpoint 参数。
- `scripts_run/run_train.sh` 当前未注释的命令实际是 **Climb Stage 2**，Object recipes 是注释示例。不能直接执行该脚本来宣称启动 Object Stage 1。
- clean 复现真正使用 `R/run_stage1.sh`，直接调用 clean `train_agent.py`，W&B offline、TensorBoard；smoke/capacity/train 分别 64×3、8192×20、8192×15000。运行目录、logger 和资源配置属于运行编排差异。

### 2.3 复现结果与可用 checkpoint

| 已有运行 | 已读证据 | 可作出的判断 |
|---|---|---|
| `R/smoke/training/WholeBodyTracking/20261123_143920-g1_29dof_wbt_hybrid_object-locomotion` | `smoke/verification.json`；64 Track/0 Gen，3 updates，actor 参数变化 | 训练链路成功；`model_00002.pt` 是 smoke 模型 |
| `R/capacity/training/WholeBodyTracking/20261123_144154-g1_29dof_wbt_hybrid_object-locomotion` | `R/capacity/verification.json`；8192 Track/0 Gen，20 updates；GPU peak 13493 MiB | 容量验证成功，不是收敛证明；该目录 `model_00019.pt` 是已验证 checkpoint |
| `R/train/training/WholeBodyTracking/20261123_144935-g1_29dof_wbt_hybrid_object-locomotion` | 16 条 train scalar（iteration 0–15）；仅 `model_00000.pt` | 早期尝试，无收敛证据 |
| `R/train/training/WholeBodyTracking/20261124_033635-g1_29dof_wbt_hybrid_object-locomotion` | 实际 YAML、train.log、events；本轮读取 iteration 0–6153（6154条训练scalar）；`model_00000.pt`、`model_03000.pt`、新增`model_06000.pt`、滚动 `model.pt` | 日志显示明显学习；用户已确认当前训练结果通过 viewer 验证。未提供该次验证的准确 checkpoint，不将确认擅自绑定到某个编号；本轮未解包权重，不能用日志最新 iter 给滚动文件命名 |

正式运行的 resolved YAML 与 `R/upstream_default_preset.json` 逐字段比较：algo/command/observation/reward/termination/robot/simulator/randomization/curriculum/training 中，仅 learning iterations 30000→15000、track_ratio 0.5→1.0、motion_dir、scene spacing 3→4、num_envs 4096→8192 不同；其余对应字段一致。正式运行 20261124 的最近 100 次 scalar 均值：

| Metric | 数值 | 解释 |
|---|---:|---|
| `Train/mean_reward` | 39.6821 | 比初期负回报有明显改善，不等于任务成功率 |
| `Train/mean_episode_length` | 417.0024 steps，约 8.340 s | control dt=0.02 s；并不证明连续完成原 clip |
| `Policy/mean_noise_std` | 0.30722 | 从初始化 1.0 学习下降 |
| `Termination/Episode/bad_tracking` | 0.27042 | 已结束 episode 统计，仍有明显失败 |
| `Termination/Episode/timeout` | 0.71393 | 超时不等于完整物理任务成功 |
| `Env/motion/error_body_pos` | 0.05310 m | 训练分布上的平均误差 |

`RewardManager` 将 episode 分项按固定 `max_episode_length_s` 归一化，raw 分项仍是累计 raw reward；不能把例如 raw object reward 的本轮约 38.23 解读为单步指数 reward 超过 1。原 `MotionCommand.step()` 在 clip end 会 reset 物理状态但不结束 episode，因此长 episode/timeout 更不能代替无 reset 的完整动作评估。

新增两个已有evaluation目录：

- `/home/han/pyx/HMI_clean/logs/WholeBodyTracking/20261124_093007-g1_29dof_wbt_hybrid_object-eval`
- `/home/han/pyx/HMI_clean/logs/WholeBodyTracking/20261124_094212-g1_29dof_wbt_hybrid_object-eval`

两者实际配置均为2 envs、Track1.0、原movebox、max_eval_steps10000、physics200/decimation4、evaluation horizon100000s、关闭video recording；各只有YAML和88-byte event文件，event tags无scalar/tensor/图像记录。这些文件本身只能证明evaluation初始化，未保存完成指标；**用户另行确认了当前训练结果的 viewer 验证成功**，该观察作为独立证据纳入。YAML中的 `training.checkpoint=None` 也不能反推“没有加载checkpoint”：`U/eval_agent.py:run_eval_with_tyro` 从独立 `CheckpointConfig.checkpoint` 加载，需启动参数/加载日志才能把此次eval与某个模型对应。本轮没有运行该入口。

`R/README.txt` 的“formal training has not been started”已落后于实际产物。用户于本轮明确补充：“当前训练我已经viewer验证过时成功的”。据此将结论更新为：**原版 Object Stage 1 已通过用户 viewer 人工验证，原版复现成功（人工观察层面）**。日志提供学习进展的补充证据。未保存的准确 checkpoint 对应关系、多起点成功率和误差分布属于后续对照记录的完善项，不作为否定该复现结果或开始最小实现的阻塞。此确认仅针对原版训练，不扩展到尚未训练验收的 Loaded-4D。

旧最新恢复运行：`A/unified_hoi_hmi_stage1/formal_recovery_after_probe_20261123_2050/training/logs/WholeBodyTracking/20261123_125742-unified_hoi_stage1-locomotion/training_status.json` 当前记录 `interrupted`、2232 completed updates、actor/critic 各 44640 optimizer steps；同目录存在 `model.pt`。旧报告的 2078/running 是历史快照，不能代替现在的 status。该 checkpoint 的输入/action/任务协议不适合作为新 Loaded-4D 初始化。

## 3. Upstream vs Previous Implementation

### 3.1 架构与学习条件比较

| 项目 | Original HMI（实际 Object hybrid preset） | Previous modified implementation | Proposed minimal Loaded-4D |
|---|---|---|---|
| 任务 | object endpoint-conditioned Track+Gen，含完整取放 reference | Carry → FullBody joint → Unified HOI 多代；最后含 pickup/carry/putdown | 已持箱；纯四维 command；一个共享 policy |
| Actor | 111D，history 总开关关闭 | 116D×3=348D；Unified 119D×3=357D，额外 3D intent | 删除 endpoint，补实际状态；候选 113D，推导见 §5；默认无 history |
| Critic | 316D；保留 Track privileged reference，Gen 对 reference terms 清零 | joint 184D；Unified 曾187D，修复后207D（119+68+20 support） | 原 critic 仅必要调整，候选315D；两阶段一致 |
| Network/PPO | MLP 512/256/128 ELU，29D action；原 PPO | 基础 PPO loss 保留，但多层 Carry/FullBody/Unified subclass 改编排/恢复/导出；历史 minibatches 曾128 | 原 PPO/MLP；不复制 subclass |
| Action | `q_target=q_default+scale*a`，clip ±100，非 ±1 | Stage1/Unified 默认 `joint_delta_v1`，center 为每 policy step 实测 q | 保留原固定中心；delta deferred |
| PD | 原 Kp/Kd，scale=0.25×effort/Kp，torque clipping | FullBody Kp×4、Kd×2，踝 Kp≥250、Kd≥12；更早 MVP 踝300/15 | 原 gains/scales/limits，不先提高增益 |
| Physics/control | 200 Hz physics、decimation4、50 Hz policy | FullBody 1000 Hz/20/50 Hz；solver16/8；固定 P0 physics | 先200/4/50；1 ms单独可选对照 |
| 初始化 | noisy RSI；soft joint-limit clipping；reset_all 含 zero-action physics step | hard-limit validation、exact reset、无初始空转、post-action reference fix | 移植这些局部语义，保留 loader/sampler |
| Clip end | resample 并物理 reset，episode 未必终止 | Track clip timeout；Gen 不因 reference end reset | Track 显式 truncation；Gen clip end 无物理效应 |
| Reward | 8 tracking terms；Gen 每项常数0.5；endpoint once bonus；3 regularizers | 原 tracking + 多种 grasp/slip/style/phase reward，移除 Gen 常数 | Track 主体不变；Gen 增3 command rewards、最小持箱约束；保留并单列原常数 |
| Termination | Track bad tracking；Gen 仍依赖 reference torso/object；low root | 更严格接触规则→support/release 状态机等多次变化 | Track 原 tracking＋clip truncation；Gen 仅实际物理失败、episode timeout |
| Noise | init std1.0、可学习；上游 Stage2 reset0.45 | init0.3可学习；诊断中增长至0.774/更高 | 不抄旧0.3或固定std；先保留上游1.0；Stage2默认继承已学std |
| Data | 原 movebox 两条完整序列及 terminal hold | joint3条、Unified6条及phase annotations | 明确只加载指定3条，无全序列/镜像的隐式混入 |

`O/agents/ppo/ppo.py` 与上游该文件的 diff 为空；不能说旧实现“重写了 PPO 算法”。主要污染对照的是观测、action、PD、reward、reset/termination、sampler/编排和 physics 的共同变化。历史 `P/STAGE1_SUPPORT_REPAIR_20261122.md` 记录实际 minibatches128→4 修复；这不是当前 preset 仍使用128的证据。

### 3.2 上游关键调用链与语义

1. `train_agent.py` 构建 typed config → `WholeBodyTrackingManager` → setup managers → `PPO.setup()` → `PPO.load()`（若配置 checkpoint）→ `PPO.learn()`。experiment 配置来自当前 CLI；训练加载权重不等同于 evaluation 从 checkpoint 配置构造环境。
2. `BaseTask.step()` → action `process_actions()` → 每 physics substep `JointPositionActionTerm.apply_actions()`/PD → `_post_physics_step()`。IsaacSim WBT 路径先刷新物理状态，再 termination、reward、reset，随后 command manager step 和 observation。
3. `MotionLoader` 读取 Holosoma NPZ：剥离 joint_pos 前7个 root DOF、joint_vel 前6个 root velocity；NPZ quaternion wxyz→runtime xyzw；按 names 重排。`MultiMotionLoader` 扫目录 NPZ并检查共同 fps。它每 policy step 索引加1，不自动按任意 fps 重采样；新任务必须 assert `motion.fps*env.dt==1`。
4. `HybridMotionCommand._build_fixed_env_partition()` 按 `track_ratio` 与 partition seed 固定分组，reset 不将 Track 切成 Gen。原 sampler 有 track/gen groups、失败 EMA、uniform mixing，可继续使用。默认先不新增 family scheduler。
5. `_compute_goal_time_steps()` 选 motion end−1；reset 时 `_assign_goal_time_steps()` 更新 goal version；`_refresh_active_goals()` 生成 exact/active endpoint，Gen 加位置/RPY噪声。Actor/Critic 的 `obj_goal_pos_b/obj_goal_ori_b` 观察它；`HybridGoalObjectReachedOnce` 以阈值0.2m/30°、bonus3给予 Gen 一次性 reward。Loaded preset 必须同时移除观测、reward、goal noise curriculum，不能只删 Actor 两项。
6. `hybrid_motion_command`、`hybrid_motion_ref_*`、`hybrid_obj_ref_*` 在 Gen rows 清零；实际 robot/body/object terms 保留。`HybridBadTracking` 的 Gen z/xyz/object checks 是对当前 reference 的误差，不能误称全部是 endpoint termination；它们也必须禁用。
7. `HybridGenNoiseCurriculum`（`managers/curriculum/terms/wbt.py`）控制 endpoint 噪声，不能直接当四维 command curriculum。`AverageEpisodeLengthTracker`、原失败采样器和统计可保留。原 `managers/command/terms/locomotion.py:LocomotionCommand` 直接采样 `lin_vel_x/lin_vel_y/ang_vel_yaw` 三个值，并非目标yaw-angle接口；已有stand概率和按时间重采样。即使 `command_dim` 设4，`_resample()`仍只赋前三列，也没有pelvis-height、strict heading-rate测量、Track/Gen隔离或连续插值的完整配置。因此可借用小段采样逻辑，不能只改command_dim就完成新任务。

### 3.3 原版真实观测布局

证据：`U/config_values/wbt/g1/observation.py`、`U/managers/observation/terms/wbt.py`、`ObservationManager`，以及 `R/smoke/console.log`/`capacity/console.log` 实际 Linear 输入111/316。**拼接按 term name 字母序，不是 preset 字典书写顺序**。

| Actor 实际顺序 | 维度 | 语义 |
|---|---:|---|
| actions |29|上一 policy action|
| base_ang_vel |3|pelvis/root full body frame angular velocity|
| dof_pos |29|q−q_default|
| dof_vel |29|原始 joint velocity|
| obj_goal_ori_b |6|endpoint orientation 的6D表示|
| obj_goal_pos_b |3|endpoint position|
| obj_lin_vel_b |3|箱体 world velocity 旋转到 torso reference frame；不是箱体相对 torso 的速度差|
| obj_ori_b |6|实际箱体相对 torso 的姿态|
| obj_pos_b |3|实际箱体相对 torso 的位置|
| **总计** |**111**|无完整 reference pose；但也无实际 root linear velocity、gravity、height|

Critic 实际字母序：`actions29, base_ang_vel3, base_lin_vel3, dof_pos29, dof_vel29, motion_command58, motion_ref_ori_b6, motion_ref_pos_b3, obj_goal_ori_b6, obj_goal_pos_b3, obj_lin_vel_b3, obj_ori_b6, obj_pos_b3, obj_ref_ori_b6, obj_ref_pos_b3, robot_body_ori_b84, robot_body_pos_b42`，合计316。14个 tracked bodies；`motion_command=reference joint_pos29+joint_vel29`，不是新任务四维 command。

两组 `history_length=1`，`ObservationManagerCfg.history_augmentation=False`。可选 history 定义包含6项99D，duration1s、stride10 control steps→offsets50/40/30/20/10（源码返回从旧到新的顺序），启用时额外495D；相应606/811D，而非默认。Actor 的 ang_vel/q/qd noise 为0.10/0.005/0.25，其他现有物理 terms noise0；Critic enable_noise=False。PPO empirical normalizers 单独作用于两组，clip observations100。Gen privileged reference的mask发生在raw terms；归一化后 `(0−mean)/std` 通常不是0，正确验收应检查raw reference为0且不随Gen隐藏reference改变，而非强迫normalized slice为0。

## 4. Reuse / Remove / Modify Analysis

| Component | Existing implementation | 决策 | 原因 | 证据 |
|---|---|---|---|---|
| 三条NPZ及hash/provenance | `P/datasets/unified_hoi_v1`，P0源副本 | Reuse | 格式/name一致；按物理验证范围标注，不重新生成 | §5数据检查；manifest及validation JSON |
| HMI motion loader/失败采样器 | `U/managers/command/terms/wbt.py` | Reuse | 多motion与names映射已有；只需三文件视图和fps断言 | `MotionLoader/MultiMotionLoader/AdaptiveTimestepsSampler` |
| 四维任务接口 | `O/carry/command.py:CarryCommand.set_command` | Modify局部复用 | finite/shape/Gen-only/no physical write可复用；旧range不能全用 | `FullBodyCommand.set_command`及sampling验证明确拒绝vy/yaw/倒退 |
| command interpolation | `O/carry/fullbody_sampling.py:transition` | Reuse数学，改配置 | cubic smoothstep及height slew-rate限制可用；不迁移mode/cycle状态 | 同文件 `sample/validate_commands` 仅vx≥0且vy=yaw=0 |
| reference extraction | `CarryCommand._update_track_commands` | Modify | heading vx/vy与pelvis来源正确；omega_world_z不等于heading rate | 本轮差值0.167rad/s；`rewards.task_errors`同错 |
| exact reset hook | `O/managers/command/terms/wbt.py`、`Stage1TrackCommand` | Modify/局部迁移 | 合法pose被95%软限位改动 | 本轮最大裁剪0.08062rad；历史12组真实reset实测q误差0 |
| reference/post-action clock | `Stage1TrackCommand.advance_reference_to_post_action`、`Stage1CarryEnv` | Modify/局部迁移 | 新命令须与reward reference严格对齐，且Gen不teleport | 原 `MotionCommand.step`/`BaseTask._post_physics_step`；历史边界测试 |
| reset_all无空转 | `Stage1CarryEnv.reset_all` | Reuse语义 | 原reset_all先zero-action推进20ms；首个Actor看不到精确起态 | 原 `BaseTask.reset_all` 源码 |
| 纯物理测量与几何箱底 | `fullbody_rewards.box_bottom/physical_metrics` | Reuse纯函数思路，去依赖 | 应按实际箱体旋转+mesh顶点求clearance，不能假设局部z朝上 | 旧axis诊断和物理审计 |
| pair-specific hand/box contacts | `fullbody_simulator.IsaacSim` | Defer训练；可选评估 | net hand force不等于hand-box contact；先不复制physics setter和friction聚合体系 | 原sensor无filter；旧sensor有Object/baseLink filter |
| pickup/putdown intent/support/release | `unified.py/support.py/unified_presets.py` | Remove from新任务 | 任务从已持箱开始，没有接管/释放阶段 | 当前357D Actor和20D critic support |
|旧complex grasp/style奖励|`fullbody_rewards/unified.py`|Remove默认|额外假设侧夹法向、摩擦承重、phase，且改变reward量级|旧grasp2、slip0.5、nonhand−2及phase rewards|
|原endpoint reward/curriculum|`HybridGoalObjectReachedOnce/HybridGenNoiseCurriculum`|Remove新preset|四维外部任务无endpoint|上游goal调用链|
|PPO/MLP/normalizer|`U/agents/ppo/ppo.py`|Reuse|不需新算法；checkpoint常规状态已支持|§6 transfer|
|旧delta action/高PD|`carry/action.py/fullbody_presets.py`|Defer|是学习条件改变，不是无损修复；改善短期接触不等于必需|32-update fixed/delta对照仍未验收|
|history348/357D、support critic|`observations.py/unified.py`|Remove默认|原HMI无history；新增内容与旧任务耦合|§3 actual dimensions|
|atomic save/安全中断/plain logging|`checkpoints.py/unified_ppo.py`|Defer独立运行维护|有工程价值，但不是Loaded算法必要组件，不移植整个PPO|Rich stall记录与上游当前可训练事实|

**旧失败证据的正确边界：**

- `P/STAGE1_REPAIR_VALIDATION.md`：32 updates单种子比较中，reset/time+delta 的Forward累计双手接触率19.69%，fixed4.57%；delta object误差有取舍；最终短模型三clip各0/2。可以支持“delta改变短期学习行为”，不能证明“原action导致全部失败”或“delta是最终方案”。旧±84的动作偏置与旧P0默认腕姿态及4×Kp后小scale有关，不能原样归因于clean HMI。
- `P/STAGE1_LEARNING_REGRESSION_20261123.md` 与 `experiments/unified_hoi/learning_regression_20261123/causal_summary.json`：1300–1399→1800–1899 mean reward5.87→1.59、std0.460→0.688、完整序列接触62.1%→38.5%。匹配新mean网络只换旧std，loaded-walk接触66.9%→78.7%；新mean关闭噪声仍比旧mean差（82.4% vs91.5%）。支持噪声有因果贡献，**均值策略退化根因未隔离**。
- 改过的PD、1ms、reset、reward、observation、minibatches均是已确认学习条件差异，不能单凭相关性判为失败原因；新最小基线不一次性复制它们。

## 5. Proposed Stage 1

### 5.1 三条reference与轻量数据检查

建议权威训练输入均取以下目录的指定三文件（不是加载整个6文件目录）：

`/home/han/pyx/g1_carry_baseline_new/HMI/datasets/unified_hoi_v1/motions/`

| 文件 | SHA256（本轮重新计算） | 帧数/fps/时间跨度 | 来源及验证边界 |
|---|---|---|---|
| `loaded_locomotion.npz` | `fe03f9072560cc75096c5a0cbcd1ae781a1fe9607c23fa8781ed257ebed98550` |110/50/2.18s|上游reframed原motion `[95,205)`；root/joints/object保留float32，body FK和velocity重算；kinematic target，未获完整物理证书|
| `continuous_grasped_squat_stand.npz` | `412681d73670e24c4a32401970710513ba9e3c2dd8b3205e2bf4c4a6a4f33a90` |351/50/7.00s|P0合成12cm侧夹蹲起；byte-identical validated copy；P0 controller验证，不是PPO|
| `repeated_grasped_squat_stand.npz` | `d6238a97fa785d94cdf9579f82c002f2b088c34267ba35465e0ed350932ca835` |651/50/13.00s|P0重复蹲起；同上|

时间跨度采用 `(N−1)/fps`，不混用 `N/fps`。

蹲起源位于 `A/carry_reference_synthesized/validated_motions/`，同名 `.validation.json` 指向 `evidence/squat_12cm_001_data/validation.json` 和 `evidence/repeated_12cm_001_data/validation.json`。本轮重新核实源NPZ hash与训练副本一致；`P/datasets/fullbody_carry_track_v2/motions/` 内两条蹲起也一致。P0记录分别为9s/15s测试（含其控制流程），box tracking最大约4.83/4.84mm、pelvis excursion约0.1214m；不等于新policy在2种grasp间连续切换成功。

`P/datasets/unified_hoi_v1/manifest.json` 明确 locomotion `physics_validation=unvalidated`，`audit/physics_audit.json` 报告早期reference position probes、bounded policy evaluations均未通过。canonical FK最大误差约1.38e−7m只证明FK一致，不是物理修复。`A/stage1_repair_validation/geometry_sampled.json`、`source_crop_comparison.json` 显示旧forward crop来自相同上游区间且抓持几何不同；该旧crop不应与本次loaded NPZ混用文件名。

三文件实际schema完全一致：`fps(1)`，`joint_names(29)`，`body_names(34)`，`joint_pos(N,36)`，`joint_vel(N,35)`，body pos/lin_vel/ang_vel `(N,34,3)`，body quaternion `(N,34,4)`，object pos/lin_vel/ang_vel `(N,3)`、object quaternion `(N,4)`。数值float32；names Unicode；全部数值finite。NPZ四元数wxyz，pelvis为body index0；root position/linear velocity与body pelvis对应字段差0。pelvis quaternion norm最大误差约6e−8；object quaternion norm最大误差行走1.19e−7、蹲起5.96e−8。Object angular speed最大值行走约1.18638rad/s、两条蹲起约0.007486rad/s。

本轮直接对比loaded_locomotion与clean上游 `[95,205)`：完整joint_pos、object_pos、object_quat最大差分别约5.85e−8、5.85e−8、2.94e−8，为float32舍入量级。clean与旧robot的 `main_mesh_collision_halfspherehand.urdf`、`meshes/half_sphere.obj` 文件hash一致。运动学来源一致和资产一致并不消除控制器/physics差异。

三文件joint_names顺序一致：左腿6（hip_pitch/roll/yaw、knee、ankle_pitch/roll），右腿6同序，waist_yaw/roll/pitch，左臂7（shoulder_pitch/roll/yaw、elbow、wrist_roll/pitch/yaw），右臂7同序；都带 `_joint` 后缀。未来仍必须走loader名称映射，不假设PhysX顺序相同。

以参考pelvis heading定义计算的分布如下；yaw_rate列使用 `q, omega_world` 的严格heading导数，见§5.2。

| Clip | vx min/max m/s | vy min/max m/s | yaw_rate min/max rad/s | pelvis h min/max m |
|---|---:|---:|---:|---:|
| loaded_locomotion |0.29338 / 0.78708|−0.08192 / 0.23791|−2.12396 / 1.49753|0.72310 / 0.74175|
| continuous squat |−0.01552 / 0.01551|−0.000669 / 0.000670|−0.002716 / 0.002716|0.60688 / 0.72688|
| repeated squat |同single|同single|同single|同single|

**联合覆盖而非各维范围的笛卡尔积：**1112帧中，`h<0.70且vx>0.1`、`vx<−0.1`、`|vy|>0.1且|vx|<0.05`、`|yaw_rate|>0.3且|vx|<0.05` 的帧数均为0。高yaw_rate是这条短行走中pelvis heading摆动的瞬时值，不应直接据此设持续转弯command上限。高位前行与近零平移蹲起是两个窄分布；没有已验证walk–squat–walk连接轨迹。

Object方面：行走box z约0.78732–0.84492m，最大线速度0.65409m/s；两条蹲起box z约0.72114–0.84000m、最大线速度0.11263m/s。box相对pelvis的world平移均值，行走约`[0.3196,0.0066,0.0867]m`，蹲起约`[0.4298,−0.0020,0.1137]m`；两类腕姿态明显不同：行走左roll/pitch/yaw均值约`[−0.100,0.273,0.049]rad`，蹲起约`[1.672,−1.054,1.614]rad`，右侧也不同。不要固定一套P0 hand pose来惩罚全部reference。

速度一致性：行走root linear velocity与本地截断序列有限差分最大差约0.02064m/s，蹲起约1.3e−6m/s；严格yaw-rate与unwrapped-yaw中心差分内部最大差分别约0.00525、1e−7rad/s。截断端点和离散导数会造成差异，禁止跨clip求导或静默重写NPZ。本轮检查不替代后续 angular velocity/全部body FK运行时验收。

以clean robot URDF限位轻量核查：三条reference无硬限位越界；95%软限位会让行走部分ankle_roll/waist_pitch最大改变0.01309rad，蹲起waist_pitch/两腕yaw最大改变0.08062rad。必须保留软限位用于原regularization/其他任务，Loaded reset单独按硬限位校验后原样设置。

### 5.2 Command严格定义、时间和坐标

- task root指定 `pelvis`；上游tracking anchor为 `torso_link`。**不把tracking anchor改成pelvis**：这会同时改变robot tracking与box observations。四维command单独从pelvis提取。
- 令 `u=R(q_pelvis)e_x`，`psi=atan2(u_y,u_x)`。`[vx,vy]=(Rz(psi)^T v_world)[:2]`。参考与实测分别使用各自heading；这正是heading-frame指令语义，不在参考heading下旋转实际速度。
- 对world angular velocity `omega`，`u_dot=omega×u`，严格 `yaw_rate=(u_x*u_dot_y-u_y*u_dot_x)/(u_x²+u_y²)`。Reference、actual reward和诊断共用这个纯函数；再与每clip `unwrap(psi)` 的差分交叉验证。需对heading投影近零做有界保护和姿态失败处理，不能仅靠除法clamp掩盖倒置/奇异姿态。
- `h=pelvis_z−ground_z(pelvis_xy)`；首版plane时actual ground为env origin z，reference为其本地plane z=0。不是box z，不是torso z。暂不新增terrain estimation系统。
- 旧code的vx/vy和height总体方向正确，但 `root_ang_vel_w[:,2]` / `root[:,12]` 两处都要替换；行走中严格heading-rate与omega_z最大差0.16703rad/s，不能因为command与reward同错便认定语义正确。

**推荐离散约定（Stage1/2统一）：**

```text
reset: physical x(k), reference r(k), command g(k), observation o(k)
pre-physics: snapshot applied_command = g(k); a(k) = policy(o(k), g(k))
physics holds action through one 20ms control interval
post-physics: actual x(k+1); Track advances exactly once to r(k+1)
prepare g(k+1) for next observation (no physical write)
tracking reward/termination use x(k+1), r(k+1)
Gen command reward uses applied_command = g(k)
capture terminal/final observation with x(k+1), r(k+1), g(k+1)
reset only terminal envs; post-reset callback refreshes without advancing again
```

Track本阶段不额外叠加command reward，reference监督r(k+1)，actor读取g(k)；这是明确的一步控制约定，不是未记录的时间错位。若未来Track也加command reward，同样用applied buffer g(k)。Gen更新/插值先后必须使Actor看到的command与本步reward目标一致；不能在physics后先重采样，再用新goal评价旧action。

reset后第一次callback只刷新reference与command；reward/termination/log重复调用不得再次advance。用单调control-step计数/显式step token保证一次推进，不使用PPO可随机初始化的episode_length作为唯一时钟。首版可配置 `init_at_random_ep_len=False`，记录这是初始化控制的明确差异，避免照搬旧delta clock对随机episode buffer报错。

Track到末帧：先评价末帧，再 `is_timeout=True` 截断并经正常reset路径换clip；final observation必须在reset前捕获。**如果同一步发生真实物理失败或bad tracking，failure优先，必须清除该env的timeout bootstrap标志。** `TerminationManager.check()`原本独立OR合并timeout与failure，`PPO._rollout_step()`只要 `infos["time_outs"]` 为真就bootstrap，因此这项互斥应在新Loaded env小范围处理并测试，不改PPO。clip结束采样计数只能记一次正常截断，不同时重复调用clip-end与episode-reset的sampler登记。Gen不推进训练reference到end，不因其结束触发reset，reference只用于合法初始持箱状态。严禁沿用基类clip-end物理重置分支。

### 5.3 最小观测方案与信息不足边界

默认从上游两组terms出发：删 `obj_goal_pos_b/obj_goal_ori_b`（9D），加 `task_command`（4D）；Actor补上游已有 `base_lin_vel`（3D）和 `projected_gravity`（3D），两组补 `pelvis_height`（1D）；Critic仅补缺失的gravity3D和height1D，已有base_lin_vel不重复加。保持actual object pos/orientation/velocity的上游torso frame语义。首版不新增object angular velocity；它是当前实际状态的潜在缺失项，若出现旋转/失握控制的可观测性问题，优先比较增加3D实际object angular velocity与现有history，不引入reference pose。

候选Actor字母序：

`actions29, base_ang_vel3, base_lin_vel3, dof_pos29, dof_vel29, obj_lin_vel_b3, obj_ori_b6, obj_pos_b3, pelvis_height1, projected_gravity3, task_command4` → **113D**。

候选Critic：原316−9+4+1+3=**315D**。字母序是原critic删除goal两项，再在body terms前加入 `pelvis_height, projected_gravity`，最后 `task_command`。`motion_command58`仍保留为Track privileged reference并在Gen置零，不能复用这个字段名覆盖成4D。actual body的42/84维继续保留。Actor与Critic不要求彼此同维，但两阶段各自协议完全相同。

为何不默认再加actual yaw_rate/heading velocity：`base_lin_vel`是完整base frame速度，`projected_gravity`确定roll/pitch后可变换到heading frame；`base_ang_vel`加roll/pitch可恢复严格yaw derivative。这避免重复已有角速度。reward和日志直接计算严格heading velocity/yaw_rate，不依靠网络推导。若实验证明显式heading velocity或yaw_rate对学习必要，作为**独立观测升级**添加/替换，重新固定两阶段schema；不能训练到Stage2才改输入。

现有Actor缺gravity，不能只机械地删9加4；absolute height和base tilt对loaded squat与保持平衡有明确作用。113/315是上述方案的计算结果，实施时仍需输出每term维度、slice、噪声、normalizer shape进行校验，不用硬编码数字替代审计。

新增 `task_command/pelvis_height/projected_gravity/base_lin_vel` 首版scale=1、noise=0，保留SI语义；现有Actor noise沿用上游，Critic无noise。两阶段normalizer的字段顺序、shape和更新策略相同。

默认history_length=1、history_augmentation=False。相同height在下蹲/上升均会出现，但实际q/qd/root velocity通常区分运动方向；单帧command加实际状态仍不能在“静止hold结束即将反向”时预测未来，也不能唯一指定同一command下所有参考grasp/步态。先看单技能和joint的条件歧义误差：若只在reversal/dwell/两类重叠区域系统性失败，再试上游history；若实测history仍无法解释未来分支，才考虑极小command lookahead（例如下一100ms height，Gen由同一插值器给出，外部无未来输入则明确hold假设）。不默认添加phase/ID/full pose，也不把三条参考可唯一从4D恢复当事实。

### 5.4 Data、reset、reward、training配置

1. 三个诊断preset/CLI选择（locomotion-only、squat-only、joint）共用同一个Loaded command/env/observation/action。未来通过只含指定文件的只读symlink view或byte-identical staging配置 `motion_dir`；locomotion可直接 `motion_file`。不把Unified整个目录传入loader。目录视图不是新motion generator。
2. Stage1 `track_ratio=1.0`，默认原adaptive sampling，frame0 probability先保留0.05；按clip报告实际采样份额，避免把长重复clip的帧数优势误当能力均衡。仅在确有某clip饿死时再加family priors，不先抄旧固定family/parity系统。
3. 新preset显式保留WBT profile的self-collision=True、effort/Kp action scales=True和init root位置配置；`U/envs/wbt/config_utils.py:apply_wbt_robot_defaults()`只对精确原env_class字符串生效，新subclass不会自动得到二次覆盖。已有Object preset含这些值，直接继承并断言即可，不必修改config_utils。reset只选择已loaded的三clip；无pickup/putdown。关闭initial pose/velocity/object noise以核对精确reference，校验硬限位后同步机器人和自由box pose+vel；不设置kinematic attachment、weld、gantry或每步写object state。关节极限噪声不通过时应失败而非裁剪。
4. 禁用default pose prepend/append、terminal hold，显式 `load_fixed_scene_from_motion=False`（三条NPZ无fixed-scene metadata，动态box仍由原robot.object配置生成）。行走末态非静止，不应凭空补1秒zero-velocity hold。明确Track clip结束是截断。episode先设15s以容纳13s重复蹲起全片；这只改变horizon，不改physics/control频率，且按秒统计分项避免10s/15s日志标度混淆。
5. reward保留原8项：robot anchor position/rotation权重0.5/0.5，relative body position/rotation、body linear/angular velocity、object position/rotation各1；对应sigma0.3、0.4、0.3、0.4、1.0、3.14、0.3、0.4。保留action_rate−0.1、joint_limit−10、undesired_contact−0.1的函数和权重；joint-limit margin按下一项局部调整。删除goal bonus。Stage1不先叠加四维command reward或旧grasp/slip奖励。
6. reset软限位修复与reward软限位是两件事。本轮按clean URDF及原 `limits_dof_pos` 公式量化：`soft_dof_pos_limit=0.9` 时，理想行走姿态 raw limit penalty均值0.01907，蹲起约0.37459；权重−10、dt0.02后分别为−0.00381和−0.07492/step。后者约抵消理想8项tracking正奖励上限0.14的一半，形成明确的目标冲突。推荐**仅Loaded preset将soft_dof_pos_limit设1.0**，保留原函数、权重及硬限位失败；运行时PhysX限位必须再与URDF核对。此配置修正不能证明它是旧训练退化根因；不得全局改simulator软范围。
7. Track termination采用原 **HybridBadTracking的Track分支（完整xyz，不是BadTrackingZOnly）**，阈值原样；保留body_proximity。追加Track clip truncation与实际跌倒/box落地/明显分离/非finite检查。不得用P0全套毫米级或双手20ms连续接触门槛提前截断所有探索。
8. 第一版物理建议固定已有1.1kg LargeBox、原mesh/robot、gravity；用上游mass随机项 `operation=add` 参数 `[1.0,1.0]` 加到URDF0.1kg，不把 `[1.1,1.1]` 错当最终质量。IsaacLab默认重算inertia，预期diag0.022；未来运行时读取核验。material可用现有randomizer固定robot1.0/0.9、object0.6/0.6、restitution0，与数据契约对应；初版关闭push/COM/link-mass/position-bias随机化。以上是新task配置，不修改原对照preset，且**不意味着复用P0的高PD**。
9. 原PPO保持：MLP512/256/128 ELU、24 steps/env、5epochs、4minibatches、gamma0.99、lambda0.95、clip0.2、entropy0.002、adaptive KL0.01、actor/critic初始LR1e−4、AdamW weight_decay0、grad norm1、empirical normalization。探索先用原init std1.0，不同时切换旧0.3与高PD；以mean/stochastic双口径诊断决定后续单项noise调整。

## 6. Proposed Stage 2

### 6.1 Track/Gen和连续执行

Stage1通过后，从新Loaded checkpoint恢复；同一preset代码，`track_ratio=0.5`作为首版明确选择。保持固定分组机制，Track沿用Stage1 reference/reset/rewards。Gen reference只提供一次initial loaded state，不作为逐帧任务目标，不用于height范围、termination或command更新。实际episode失败/训练horizon仍可reset；**command更新和reference end绝不reset**。

`set_command([vx,vy,yaw_rate,h], env_ids)`仅更新Gen目标buffer，finite/shape/范围检查，保持全部physical tensors、reference初始身份、episode counters、action history不变；实际command以固定control dt插值。外部command目标与经过slew limit实际送给policy的command分开记录。外部持续占用时不再被随机sampler覆盖；退出外部模式只改变command来源，不reset物理。

初版单轴→组合→切换采用配置级阶段推进，仍是同一policy和实现，不先添加自适应curriculum类：

- 先loaded stand +高位小幅forward/backward、lateral、yaw及stationary height ramps，保留一定stand采样概率。
- 再两轴组合，随后允许小幅velocity/yaw与height并用；低位时先缩小速度/转速范围，不按三条NPZ min/max矩形独立采样。
- 最后缩短dwell并加入连续切换；建议起步dwell3–5s、velocity/yaw smoothstep0.8s、height rate≤0.06m/s，作为待验证初值。
- 参考候选初始范围：vx±0.15m/s、vy±0.10m/s、yaw±0.20rad/s，h先0.66–0.73m，stationary squat再逐步到约0.61m；forward恢复Stage1已学更高速度单独扩展。它们是保守探索配置，**不是已证实feasible envelope**；backward/lateral/pure-turn均是新泛化任务。

不迁移 `fullbody_sampling.sample()` 的forward-only模式限制、mandatory cycle/parity机制。可复用 `transition()` 数学；连续外部流反复更新target时须保证从当前command重新起步且不永远重启到零速度。

### 6.2 Reward、carry约束和termination

Gen command error采用§5.2同一actual measurement helper：

- `r_xy=exp(-||v_heading_xy-g_xy||²/sigma_v²)`，初始sigma_v=0.2m/s。
- `r_yaw=exp(-(psi_dot-g_yaw)²/sigma_yaw²)`，初始sigma_yaw=0.3rad/s。
- `r_h=exp(-(h-g_h)²/sigma_h²)`，初始sigma_h=0.05m。

初版各权重1，明确Gen mask；Track不重复加任务reward。三项共享“箱体仍离地、未明显分离且状态finite”的实际状态gate，避免掉箱后跑步得分。carry-maintenance先用这些条件的物理失败/诊断和原regularizers，不先增加多项法向、friction、box目标姿态奖励。box bottom按真实mesh与姿态算；允许正常抱持姿态变化，不用某条reference的box轨迹/固定局部轴约束Gen。

**原Gen常数不能未经分析删除。** `_mask_track_envs(...,gen_constant=0.5)` 对8个tracking terms都生效，权重和7，因此Gen有3.5 reward/s，50Hz时0.07/step；10s存活贡献35 undiscounted，gamma0.99的无限步上限约7。它在固定存活时长内不改变动作排序，但在有failure termination时影响存活收益和值函数；旧 `carry.rewards.track_imitation` 删除它改变了学习条件。

推荐首版保留上游8项hybrid reward及此baseline，删除endpoint bonus后加三项command rewards。Gen reference可冻结为finite reset sample，tracking函数结果在Gen被mask成常数，不能通过reference漂移改变Gen收益。日志必须把constant、command、regularization分开汇报，防止“仅站住”被当作四维控制成功。后续只有证据表明baseline压倒command任务时才单项调整，不能先删掉再将结果与原HMI总回报直接对比。

Gen termination：

- 配置将 `gen_bad_ref_pos_z_threshold`、`gen_bad_ref_pos_xyz_threshold`、`gen_bad_object_pos_z_threshold`、`gen_bad_object_ori_threshold` 设为None，真正禁用；不要设大阈值冒充解耦。
- 原min_root_height采用world z且默认0.45；Loaded交由地面相对实际physical failure处理，旧term配置 `min_root_height=-inf` 关闭其比较（不能填None触发类型错误）。首版plane、h_min约0.61m时，绝对fall floor约0.4–0.45m可作保守候选；以后若扩展深蹲需复核，不能跟随reference高度移动。
- 共同失败：明显倒置/失控、非finite、box触地/严重穿透、box与robot明显分离；阈值按mesh尺寸和允许h envelope校验。只有“真实持箱是否保持”有证据不足时才增加hand-box pair contact持久丢失检查，并采用短grace/持续时间避免单帧求解噪声。无需support transfer/release状态机。
- 训练timeout维持有限horizon；连续验收用30–60s、不因commands切换reset；若发生真实失败，记录失败并结束该trial，不将重置后的片段拼成连续成功。

上游robot net contact sensor不能单独证明hand-box接触；首版评估应至少核对自由body状态、box clearance/相对距离及视频中的持箱，若要声称双手接触/摩擦承重，则必须加可选filtered sensor并给真实contact证据。不能输出不存在的接触成功率。任务未要求只许手部侧夹，故不默认沿用旧“nonhand接触>0.5N即失败”规则。

### 6.3 Control frequency

已核实 `sim.fps=200`、`control_decimation=4`，`sim_dt=0.005s`、`env.dt=0.02s`，policy50Hz。原PD每physics substep计算，action target在policy周期内固定；reference50fps每policy step推进一帧。原增益从 `R/upstream_default_preset.json:robot.control` 与实际resolved YAML交叉核对：

| 关节组 | Kp | Kd |
|---|---:|---:|
| hip_pitch/hip_yaw/waist_yaw |40.179238471|2.557889765|
| hip_roll/knee |99.098427777|6.308801854|
| ankle_pitch/ankle_roll/waist_roll/waist_pitch |28.501246196|1.814445687|
| shoulder_pitch/roll/yaw、elbow、wrist_roll |14.250623098|0.907222843|
| wrist_pitch/wrist_yaw |16.778327481|1.068141502|

左右对称；保持 `q_target=q_default+(0.25*effort_limit/Kp)*clip(a,−100,100)` 与原torque clipping。这张表是clean基线，不是旧P0的4倍增益。

推荐先保留200/4；P0验证用1000/20不能证明1ms必需。若后续接触/solver诊断支持比较1ms：只把fps1000、decimation20（render cadence对应调整），保持policy20ms、reference50fps、action scales/default center、Kp/Kd、torque limits、PPO、reward dt全部不变。不能同时引入solver16/8、Kp×4和delta。原contact sensor `update_period=0.005`，改physics dt后不会自动成为1ms接触采样；如新增substep diagnostics，须另行注明采样周期，避免伪称完整1ms接触记录。

### 6.4 Checkpoint transfer与state

`PPO.load/save`已包含actor、critic、两个normalizers、两个optimizer、learned actor.std、iter；`BaseAlgo`通过env接口收集状态。clean `BaseTask.get_checkpoint_state()` 返回空，`WholeBodyTrackingManager`未覆盖保存完整Track/Gen partition；因此**上游目前没有Stage1 mask覆盖Stage2的既有事实**。风险来自未来新env或复制旧carry恢复逻辑。

| 状态 | Stage1→Stage2建议 |
|---|---|
| Actor、Critic参数 | 全部恢复；检查names/shapes，禁止strict=False掩盖schema差异 |
| Actor/Critic normalizer | 全部恢复；Stage2继续更新，evaluation冻结；禁止只载Actor造成归一化错位 |
| Optimizer/LR/step | 默认原 `load_optimizer=True` 恢复；reward变化后诊断value loss，但不默认清空critic/optimizer |
| Exploration std | 默认恢复learned std；不默认沿上游Object的0.45 reset，因为可能放大接触噪声；若显式选reset，复用已有参数及其std optimizer-state清理，不增新机制 |
| Iteration | 明确上游learn将num_learning_iterations解释为**追加updates**，不是目标总数；iter记录语义与completed update区别须保留，避免照抄旧wrapper的总数换算 |
| Track/Gen masks | 按Stage2新config重新构建、断言数量；绝不从checkpoint覆盖 |
| Physics、episode buffers、action history、Gen command elapsed/dwell/external flag | 重新初始化；不声称恢复PhysX接触求解器状态 |
| Sampler/RNG/curriculum | clean upstream不保存完整这些状态，默认重新初始化；保持同stage精确续训可另议。不要顺带实现庞大state持久化系统 |
| Task contract | 新env checkpoint state仅保存obs字段顺序/normalization/action/timing/data hashes等版本元数据并校验，排除stage ratio和physical state的强制恢复 |

Stage2加载后的必须检查：Track/Gen数量匹配新ratio；Gen raw reference项mask为0、actual/command正常保留（normalized reference无需为0）；Actor同一输入在load前后输出一致；command源仅Gen变化；randomized episode buffer不会破坏reference clock。原111/316或旧357/207模型均不与新113/315直接兼容，新Stage1应从头训练。

## 7. File-level Minimal Modification Plan

**本节保留经审查的文件级设计；用户授权实施后，所列新增文件已在 `feature/loaded-4d` 分支创建，完成状态以§10.2为准。** 原 `main` 与 `docs/loaded-4d-plan` 分支保留对照；源码工作区为 `/home/han/pyx/HMI_clean`。

### 7.1 必须修改/新增

| Actual file path / 拟新增path | Relevant class/function | Current behavior | Required change / 为什么 | Config或source | 可复用旧实现 | Validation method |
|---|---|---|---|---|---|---|
| `/home/han/pyx/HMI_clean/src/holosoma/holosoma/managers/command/terms/wbt.py`（已有） | `MotionCommand.reset`, `MotionCommand.step` | reset硬编码soft clip；step将advance、clip-end reset与relative targets刷新耦合 | 提取一个reset joint preparation hook（默认保持原clip）；提取relative-reference刷新为可调用hook（默认step仍同序调用）。Loaded subclass调用复用逻辑，无需复制整个reset/loader | Source，小范围提取；配置无法绕过hard-coded clip/step副作用 | 旧同文件 `_prepare_reset_joint_positions`；旧`CarryCommand._update_relative_reference`仅作对照 | 原Object/Climb CPU tests；默认reset输出不变；Loaded硬限位负例报错、合法q不变；原step regression；不全局改sim limits |
| `/home/han/pyx/HMI_clean/src/holosoma/holosoma/managers/command/terms/loaded_4d.py`（本轮已新增） | 拟新增 `Loaded4DCommand(HybridMotionCommand)`、heading测量pure helper、`set_command` | 现有Hybrid只有endpoint condition且clip-end reset | 独立4D buffers；严格pelvis heading-rate；Track时钟、Gen frozen reference；reset exact；本步applied command与下步published command；Gen sampler/插值；禁止endpoint metrics/curriculum。保留上游loader/sampler/partition | Source；范围限制/ratio/dwell/fps/阶段为params配置 | `CarryCommand` shape/finite/Gen-only接口，`Stage1TrackCommand` reset/time，`transition`数学。不得继承旧CarryCommand | CPU q/omega倾斜与yaw wrap、跨clip导数、reset首帧/末帧、重复callback、Gen setter无物理写、random-command isolation；fps mismatch fail-fast；loader跳过任何指定文件时失败 |
| `/home/han/pyx/HMI_clean/src/holosoma/holosoma/envs/wbt/loaded_4d.py`（本轮已新增） | 拟新增小型 `Loaded4DTrackingManager(WholeBodyTrackingManager)`；task observations、Gen reward、clip/physical termination functions | 原BaseTask reset_all空转；reward前未advance；无4D rewards和实际carry failure | 仅override reset_all、pre-physics applied-command快照、pre-termination推进/互斥timeout处理及必要checkpoint contract；薄terms读取同一command/helper，保留manager框架。physical geometry与reward计算无phase机器 | Source；数值阈值用params，不增新环境框架 | `Stage1CarryEnv.reset_all/_check_termination`、`box_bottom`公式、旧task_errors修正版本；不迁移FullBodyEnv或Simulator | step spy确认一次advance和zero hidden step；Gen越过reference长度不reset；drop gate、ground-relative height、final obs与truncation/failure互斥bootstrap；重复metrics无计时副作用 |
| `/home/han/pyx/HMI_clean/src/holosoma/holosoma/config_values/wbt/g1/loaded_4d.py`（本轮已新增） | 拟新增一个Loaded preset factory/共享配置，stage/data variant只是replace | 现有Object preset含endpoint、10s/terminal hold/noisy RSI等 | 配置移除endpoint terms/curriculum、增加obs/rewards/termination引用；显式保留WBT profile默认值，禁用fixed-scene metadata注入；Stage1/2共用schema；三data选择；固定mass/material可直接复用randomizers；Loaded joint-limit margin设1.0；horizon/DR/initial noise独立列出 | Configuration composition；写Python preset但无新算法 | 原Object preset优先；旧preset只查差异，不能继承整套 | resolved config diff allowlist；data文件数/hash；obs字母序/维度/normalizer/noise/action一致；原preset serialized内容不变 |
| `/home/han/pyx/HMI_clean/src/holosoma/holosoma/config_values/experiment.py`（已有） | `DEFAULTS`, `AnnotatedExperimentConfig` | 仅已有exp aliases | 导入/注册新Loaded alias；保留现有名字和值 | Source注册，极小改动 | 上游注册方式 | CLI解析/构造config无sim smoke；Object/Climb aliases未变 |
| `/home/han/pyx/HMI_clean/src/holosoma/holosoma/envs/wbt/tests/test_loaded_4d.py`（本轮已新增） | 拟新增协议测试 | 现有测试覆盖原hybrid但非新语义 | 合并必须的CPU tests：heading、time、reset、mask、schema、command更新/transfer；不为低风险配置堆叠镜像测试 | Test only | 原command/observation/reward/termination hybrid tests；旧stage1 tests择取语义断言 | 无sim/GPU单测先行，之后实施阶段专门短Isaac验证真实reset |
| `/home/han/pyx/HMI_clean/scripts_run/evaluate_loaded_4d.py`（本轮已新增） | 拟新增薄evaluation entry，调用原加载/推理接口 | 原viewer/metrics没有4D连续执行与clip-end物理写统计 | 承载Stage1按clip/起点和Stage2 command sequence评估；记录失败和reset、mean/stochastic、tracking/command/clearance；同一decoder/normalizer | Evaluation source，非trainer | `P/scripts/carry/evaluate_fullbody.py`、`evaluate_unified_hoi.py`的统计口径，不复制phase case矩阵 | checkpoint schema错配拒绝；固定seed可复查；无command-reset；PyTorch推理先验收，ONNX按需求追加 |

将小型任务terms合并到两个runtime新文件是为限制改动面；若仓库review要求按manager分文件，可拆分而不增加行为。实际实现采用本表的 `Loaded4DCommand` 和 `Loaded4DTrackingManager`；公开命令及外部接口示例见现有 `README.md` 的 Loaded-4D 小节。

数据选择已创建三文件view（以及squat二文件view），指向§5列出的不可变NPZ；Loaded setup须断言实际加载数量、各clip长度/hash与指定集合一致。原 `MultiMotionLoader`会警告后跳过格式不兼容文件，不能把“仍加载成功一个clip”误当joint训练就绪。**不修改 `MotionLoader/MultiMotionLoader`**，不新增数据转换程序。如果实现团队希望支持任意显式file list，这是可选扩展，不属于首版必需。

### 7.2 可选修改：必须有触发条件

| 已确认相关文件/位置 | 条件与最小范围 | 不采用默认迁移的原因 | 验证 |
|---|---|---|---|
| `U/simulator/isaacsim/isaacsim.py:_setup_scene`；旧`O/carry/fullbody_simulator.py:_setup_scene/raw_contacts` | 只有需要量化真实双手接触或发现pose gate作弊时，在新task setup增加hand→box filtered ContactSensor（优先不改共享simulator）；固定配置，无mass/PD setter | net force无法归属接触物，但完整旧friction/support simulator过重 | 已知接触/无接触负例；query-only不改变physics；注明采样周期/force方向 |
| `U/managers/observation/manager.py`、新preset | reversal歧义明显才启用已有history；**不改manager源码** | 现有机制足够；不能默认抄348D/357D | 同一step history偏移、reset padding、两stage一致、normalizer shape |
| 新preset中的sim参数 | 接触稳定性证据需要时200/4→1000/20，仅这一组保持policy频率所需的联动 | 不把1ms与高PD/solver/delta一起改 | policy/reference时间相等、action target相同、reward dt不变 |
| `U/agents/ppo/ppo.py`及logging helper | 仅复现新的logging阻塞/保存丢失问题后做独立维护补丁 | clean已运行数千updates；旧Unified编排不是Loaded必要依赖 | 保存/恢复与更新数对照，不改变PPO loss/rollout |

### 7.3 暂缓或保持不变

`U/agents/ppo/ppo.py`、`agents/modules/ppo_modules.py`、`managers/action/terms/joint_control.py`、robot URDF/mesh、box URDF/mesh、IsaacLab源码/runtime、`BaseTask`、通用observation manager、motion loaders、原reward/termination functions均默认保持不变。局部hook不改变这些组件默认行为。原config_values/wbt/g1的experiment/observation/reward/termination/command/curriculum presets无需直接改写，可在新preset `replace`组合。

不迁移 `carry/action.py` 的delta、`fullbody_ppo.py/unified_ppo.py`、`support.py`、完整carry simulator/environment；不添加AMP、world model、hierarchical policy、motion generator、force adaptation或训练supervisor。

## 8. Implementation and Experiment Sequence

以下是按依赖排序的实施与实验流程；当前已完成的代码/CPU验证见§10.2。训练、真实物理验收和学习效果不因代码完成而视作通过。除发现具体失败外不追加大规模消融。

| 顺序 | 最小工作 | 通过条件 | 失败时优先诊断 |
|---|---|---|---|
| 0 | 从clean commit隔离新实现；保留用户已通过viewer验证的原版对照，保存resolved upstream config、现有run证据，并在确定模型时记录其路径/hash | 原preset不变；依赖仍绑定正确source；未来改动只在批准的工作区；不以缺量化报告要求重训原版 | import path/注册覆盖/运行时适配；不要先调学习参数 |
| 1 | 固定三NPZ hash、构建只读三文件view；实现纯measurement helper和schema | names/fps/finite/hard limits通过；倾斜姿态yaw-rate等价；坐标变换前后一致；联合command覆盖报告可复现 | quaternion顺序、pelvis vs torso、source/body indices、端点差分 |
| 2 | 仅Stage1必要command/env/preset；CPU协议测试 | reset(k)→首次obs(k)，无hidden physics；每步一次advance；末帧reward后truncation；Actor/critic schema稳定 | reference时序、sampler callback、episode random init、auto reset顺序 |
| 3 | 实施阶段短真实reset/rollout/saving smoke；先保持原action/PD/physics | 三clip多个合法起点机器人与box pose/vel与reference容差匹配；自由box无attachment；数值finite；保存/恢复输出一致；没有立即系统性bad reset | soft/hard limits、ground penetration、资产/质量、物理状态sync；随机policy不会持箱不等于代码门禁失败 |
| 4 | locomotion-only和squat-only短学习诊断（共用实现） | 数据正确且在同等预算内tracking/存活存在学习趋势；分清mean和stochastic；不要求短训就最终成功 | 先看contact几何、action saturation、目标soft penalty、noise；若两类都失败不要直接joint堆奖励 |
| 5 | 三条joint Stage1训练 | 每clip均有有效采样/学习；无单clip被平均回报掩盖；输出各项reward、clip completion与physical resets | grasp差异/条件歧义、sampling份额、reward与constraint冲突；有证据后再增history/调整单项 |
| 6 | 独立Stage1评估，frame0全片＋若干RSI起点，多seed，mean为主兼报stochastic | 建议审查门槛：每clip至少10个独立trial，≥90%完成指定区间且无drop/fall/隐藏reset；robot/object误差分位数及接触证据符合约定；重复蹲起覆盖完整13s | 不把随机末尾reset样本成功当全片成功；区分低位接触/抓持衔接、noise与mean退化 |
| 7 | Stage2 command来源、3项reward、Gen failure、transfer smoke | 全部权重/normalizer已恢复；新50/50分组生效；Gen reference置零/冻结；连续更新command不改任何physical state；超过reference长度仍连续 | checkpoint config与env-state污染、mask、applied command错位、旧clip-end分支 |
| 8 | 单轴→两轴→带高度组合，按验证达标再扩范围 | 各轴command误差达候选阈值并且真正持箱；Track能力不显著下降；组合单独通过，不能由单轴通过推定 | 将新axis覆盖不足与reference tracking问题分开；优先缩命令幅度/变化率，不放宽物理失败掩盖问题 |
| 9 | 30–60s连续command序列，多seed，含正/负vx、vy、转弯、squat/stand及可行组合 | 建议每序列≥10trial、≥90%无physical reset/drop/fall；排除过渡窗口后mean绝对误差vx/vy≤0.10m/s、yaw≤0.15rad/s、height≤0.03m，并报告p95/最差与切换超调；阈值为待审查目标 | stand-only高回报、单轴切换失败、低位可行性、实际contact vs pose gate、reference-dependent hidden termination |

通过条件中的90%和误差阈值是**计划建议，不是既有成绩**。若实际reference噪声或目标幅度使其不合理，review后调整并记录，不在看到评估结果后静默放宽。Stage1完整tracking通过不自动代表Stage2所有组合可行；Gen失败不应通过重置续接掩盖。

## 9. Risks and Open Questions

| 问题 | 分类 | 当前结论与实施前/后所需动作 |
|---|---|---|
| 三数据是否可读、同版本 | 已确认 | 指定hash、50Hz、29joint/34body一致；三文件视图避免误加载六文件 |
| 原版复现是否成功 | 用户人工确认 | 用户已用viewer验证当前训练成功，可作为原版对照；准确checkpoint与多起点量化结果尚未归档。不以这些记录缺项否定人工验证，也不推断未经测量的成功率 |
| 行走与侧夹蹲起是否能由同policy可靠连接 | 待实验 | 腕姿态和object offset差异已确认；没有真实walk-squat-walk证据。不是必须预先生成新motion的理由 |
| 参考是否在原PD下物理可跟踪 | 待实验 | P0验证使用不同控制/1ms；旧失败probe不能证明全局不可行。先原action/PD单技能诊断，禁止默认高PD/delta |
| yaw-rate与reference-time | 已确认需改 | 旧world omega_z不符合heading-rate定义；上游post-reward advancement已定位，但是否导致旧退化未证实；新reset/step契约在实施前定稿 |
| 合法reference被reset裁剪 | 已确认需改 | hard bounds可行、soft裁剪最大0.08062rad；按实际PhysX硬限位再核验 |
| soft joint-limit reward冲突 | 已确认且已量化 | 蹲起理想姿态约−0.07492/step；推荐Loaded margin设1.0、原函数/权重不变。PhysX限位仍需运行时核验；不误写为旧退化根因 |
| 初始物理契约 | 需review确认选择 | 推荐固定1.1kg、原几何、原PD、200/4；P0记录mass1.1kg不能通过只禁用mass DR得到（URDF仅0.1kg） |
| 单帧command可识别性 | 合理推断＋待实验 | 实际q/qd可区分多数升降；dwell/reversal/相近4D下多grasp未必可识别；默认不加history，明确触发条件 |
| carry-maintenance定义 | 需review确认验收范围 | 当前任务“抱住箱体”未强制双手纯侧夹/禁止躯干接触；建议以自由box离地稳定为初版条件。若必须纯双手承重，需filtered contact diagnostics及明确验收规则，不能沿用旧0.5N非手硬终止 |
| Gen可行范围 | 待实验 | 参考支持集窄；不能用独立min/max拼矩形；所有倒退/侧移/原地转/低位组合需新Gen学习和验证 |
| Stage1→2 normalizer/std | 已确认接口必须一致，最佳noise待实验 | 默认恢复全部网络/normalizer/std；旧噪声因果证据仅适用于旧训练，不能证明新方案应固定std或改entropy |
| 运行状态快照漂移 | 已确认 | clean训练日志可持续增长，旧状态已interrupted；后续实施重新读取checkpoint内部进度，不据文件存在自动继续训练 |

**是否阻止开始后续最小实现？** 没有缺仓库、缺数据或必须新框架才能解决的代码级阻塞；本文已给出可实施默认值。review仍需认可物理契约、carry成功定义和上述时序/观测协议。**是否阻止直接宣称已达标或立刻长训Stage2？是：新Stage1三clip物理验收、原PD下joint可学性以及连续Gen结果均未完成。** 不把它们转化成必须先由P0控制器复现全部reference才能写代码的额外门槛。

## 10. Final Recommended Modification Scope

推荐实施范围控制为：

- **修改2个现有文件**：`U/managers/command/terms/wbt.py` 仅提取默认行为不变的reset/relative-reference hooks；`U/config_values/experiment.py` 仅注册新preset。
- **新增3个任务源码文件**：`managers/command/terms/loaded_4d.py`、`envs/wbt/loaded_4d.py`、`config_values/wbt/g1/loaded_4d.py`（本轮已新增）。用params/replace区分Stage1/Stage2和单技能data view。
- **新增1个协议测试文件与1个薄评估入口**：路径见§7；已创建只含指定NPZ的相对symlink数据视图，不重写NPZ、不生成新motion。
- **保持不变**：PPO数学、MLP隐藏层、normalizer实现、joint-position action/PD、simulator/framework、motion loaders、上游原preset、URDF/mesh、第三方依赖。保留原始运行入口以做回归对照。
- **暂不做**：旧HOI状态机、旧PPO子类、自动训练/调度、复杂contact reward、独立curriculum系统、1ms、高PD、delta、history/lookahead、额外motion augmentation。

该范围已达到最小实现的目标：必要差异都能追溯到四维接口、真实自由箱体、reference监督和连续Gen要求。还能删减的是首版自动curriculum、训练中的精细contact shaping与ONNX专用封装；不能再删的是严格command语义、reset/time修复、Gen clip-end隔离、normalizer/输入协议一致性和真实物理评估。用户现已授权代码实施；后续review意见和验证结果继续同步到本计划及修改记录。


### 10.1 持续维护与修改记录

- 更新方式：收到新的实验事实、用户确认或 reviewer 意见后，直接修订相关章节；两个报告路径在同次更新中保持字节一致，避免分别编辑形成不同版本。
- 提交时机：一个明确的审查结论或修改批次完成并核验后提交；临时草稿可先保留未提交状态。文档提交只 stage 本报告，后续实施提交也仅包含对应任务的已核验改动。
- 每次提交记录：本文增加简短记录，commit message 使用标题加正文，列明“修改内容、修改原因、核验结果、未完成事项（如有）”。Git commit hash 从历史查询，不把本次尚未生成的 hash 写回同一个 commit。
- 授权范围：文档分支和本地 commit 已获明确授权；用户随后另行授权将当前代码与计划 push 到 `https://github.com/Ir1222/HMI-carry.git`。首次推送包含原版 `main` 和 `docs/loaded-4d-plan`，使用普通推送，不覆盖远端历史。随后用户已正式授权代码实施和结束后push；本轮执行代码与CPU验证，未启动训练。

| 记录 | 日期 | 修改内容与原因 | 核验结果 |
|---|---|---|---|
| 审查初稿（纳入Git前） | 2026-11-24 | 完成本地仓库、三NPZ、原版及旧实现审查；形成十章最小修改计划。根据用户补充，将当前原版Object Stage1更新为已通过viewer人工验证 | 本地源码/数据/已有日志只读核对；三NPZ格式与hash检查；原版成功结论注明来自用户观察，未虚构定量成功率 |
| D001：计划纳入版本管理 | 2026-11-24 | 在用户选定的 `docs/loaded-4d-plan` 分支提交同名报告副本，原路径同步保留；新增持续更新、阶段性commit及每次文字记录规则；将原Git状态明确标记为审查快照，避免与文档分支混淆 | 核对两个副本内容、十个必需章节和Git diff；本提交仅含Markdown报告，无Python/config/NPZ/checkpoint变更；未执行训练或仿真 |
| D002：明确工作路径与首次推送范围 | 2026-11-24 | 明确后续代码工作区为 `/home/han/pyx/HMI_clean`，计划继续在原路径和仓库内同步；记录用户授权将原版 `main` 与文档分支推送到 HMI-carry，便于远程审查 | 推送前 `ls-remote --heads origin` 无分支；核对origin地址、分支及双副本一致性；本提交仅修改报告。推送完成后核对远端两分支的commit与本地一致，最终结果以Git远端refs为准 |
| I001：Loaded-4D核心实现 | 2026-11-24 | 在 `feature/loaded-4d` 实现独立command、薄WBT环境和共享Stage1/2 presets；原motion仅提取2个hook；注册4个实验alias，创建5个不可变NPZ相对symlink；新增协议及checkpoint测试 | 17项新CPU测试和28项上游command/preset回归通过；Actor113D/Critic315D由真实ObservationManager计算；原PPO恢复网络/normalizer/optimizer/std测试通过；未启动GPU仿真或训练 |
| I002：评估入口、从头训练与使用说明 | 2026-11-24 | 完成薄评估工具及CPU数据检查；增加终止前快照/隐藏reset/失败trial测试；按用户要求在现有README加入从头Stage1、smoke、单技能、Stage2迁移和连续评估命令；同步计划 | 最终95项CPU测试通过（20项Loaded测试＋75项上游回归）；真实PPO constructor/setup确认Stage1无checkpoint也可初始化、iter0/空optimizer/新normalizers/std1.0；README五条训练命令实际Tyro解析通过；三NPZ hash/fps/联合分布复核通过；未运行仿真或训练 |
| I003：修复原版与Loaded-4D共用Viser入口 | 2026-11-25 | 用户反馈网页无法打开，并确认原版复现评估也曾遇到相同错误。定位Isaac Sim的pip_prebundle内websockets12.0遮蔽runtime已安装15.0.1；在`eval_agent.run_eval_with_tyro`启用Viser时，于simulation setup之前导入Viser，依赖缺失提前报错；补3项启动回归和README逐motion查看命令 | CPU子进程复现原`No module named websockets.asyncio`；提前导入后通过；3项新测试＋15项原Viser测试共18项通过；注入真实Isaac bundle路径后，临时回环Viser服务HTTP200及带版本subprotocol的WebSocket收包通过；未启动Isaac/GPU仿真，未修改依赖、训练代码、reference或checkpoint。完整Isaac viewer仍待用户重试 |
| I004：修复量化评估跨trial重置及异常可见性 | 2026-11-25 | 用户运行`evaluate_loaded_4d.py --mode track --trials 10`，加载成功后仅见shutdown日志且无结果JSON。CPU复现脚本中的确定缺陷：rollout的`torch.inference_mode()`使真实TerminationManager持有inference tensor，下一trial在context外reset原地修改报错；改用上游eval同样的`torch.no_grad()`。增加逐trial进度、结果输出，并在Isaac cleanup前打印flush原始traceback；同步README | 新增Track/Gen连续trial测试先在旧代码上均复现第二trial reset的RuntimeError，修复后连同异常输出顺序测试及原Loaded测试共23项通过。未修改训练/manager实现/物理/数据/模型；未运行GPU评估。本次用户日志未包含原始traceback，因此不声称已排除其他运行时问题，真实30trial结果待重新执行 |

### 10.2 实施进度与可执行入口

当前实现分支：`feature/loaded-4d`，从已推送文档版本派生。`main` 仍为 `c353731` 原版代码。

已实现：§7的两个共享文件小改动、三个任务模块与协议测试。三个任务模块采用计划中给出的类名；原PPO、MLP、PD、action、motion loaders、simulator和原preset未改。`scripts_run/evaluate_loaded_4d.py` 已完成，CPU测试覆盖首次失败停止、隐藏reset检测及终止前快照；真实checkpoint的PhysX评估尚未运行。

2026-11-25追加：用户已自行启动joint Stage1，run为`/home/han/pyx/HMI_clean_runs/loaded_4d/stage1/WholeBodyTracking/20261124_105123-g1_loaded_4d_stage1_joint-locomotion`。保存配置与train.log确认加载3条motion；末尾`-locomotion`不代表单技能。已有编号checkpoint至`model_06000.pt`（本次只读快照），用户评估日志确认113D/315D模型加载成功；viewer依赖失败不构成训练失败或reference初始化错误的证据。评估配置的`motion_id=1/local_frame=0`确为行走起点。I003修复共用入口，单独补充CPU启动测试；Stage1物理效果仍须实际viewer及量化评估确认。

- Runtime默认：200Hz physics / decimation4 / 50Hz policy，原固定中心action；Stage1全Track，Stage2固定50/50；同一113D/315D observation schema。
- Gen随机目标用0.8s smoothstep、height限速0.06m/s；外部每帧重定向使用连续限速，避免重启smoothstep造成停滞。外部目标与本步applied command分开保存。
- CPU测试已验证：严格yaw-rate、精确reset、单调step token、Gen越过clip末帧无physical write、外部Gen隔离、实际持箱gate、失败优先于timeout、final observation在reset前捕获、checkpoint迁移保留新分组。
- `datasets/loaded_4d/joint/` 中3条和 `squat/` 中2条是相对symlink，指向工作区已有旧仓库的指定NPZ；没有复制或修改NPZ。新机器须提供这些原数据并保持布局，或用CLI覆盖 `motion_dir/motion_file` 指向相同hash的数据视图；不自动下载数据。
- 本轮尚未运行PhysX reset/rollout、短训练、joint训练或Stage2训练。物理阈值、真实接触、1.1kg/inertia和原PD下可学性仍需计划中的后续实验验收。


最终验证与运行入口：

1. `README.md` 的 **Loaded-4D：当前实现与训练指令** 小节是最新操作入口；保留原版Object/Climb章节。新增README为用户追加要求的文档改动，不扩大核心源码范围。
2. Stage1从头训练：`python src/holosoma/holosoma/train_agent.py exp:g1-loaded-4d-stage1 --training.checkpoint=None`；完整环境、logger与资源参数见README。默认随机初始化网络、空optimizer、normalizer count0、iter0、std1.0、Track1.0，无自动旧模型恢复。
3. Stage1三个alias为 `exp:g1-loaded-4d-stage1`、`exp:g1-loaded-4d-stage1-locomotion`、`exp:g1-loaded-4d-stage1-squat`。Stage2用 `exp:g1-loaded-4d-stage2` 加本实现joint Stage1的checkpoint；`loaded-4d.axes` 支持 `single/pair/combined`，同一实现配置推进。
4. 数据CPU检查：`python scripts_run/evaluate_loaded_4d.py --validate-data`。评估使用 `--checkpoint ... --mode track` 或 `--mode gen --seconds 60`；支持seed/trials/RSI起点、mean/stochastic、JSON指令序列及viewer。
5. 新增20项CPU测试位于 `U/envs/wbt/tests/test_loaded_4d.py`；最终相关回归共95项通过。仓库多个目录含同名 `test_wbt_hybrid.py`，联合运行时使用 `pytest --import-mode=importlib`，无需删除缓存或改上游测试。模块编译和 `git diff --check` 通过，实际训练CLI和配置序列化往返通过。
6. 实现中的明确取舍：外部流采用可重定向限速，随机采样采用smoothstep；物理gate只证明有限的实际几何约束，不声称双手摩擦承重；继承的Gen tracking常数保留并单列日志。Stage1/2恢复时检查obs/action/timing/data契约，重建环境分组和command状态。
7. 后续实际验收仍按§8第3步起执行：小规模PhysX reset/rollout与保存恢复，再单技能/joint Stage1、独立Stage1评估、Stage2学习和连续command评估。本轮测试使用CPU数据与simulator double验证协议，**不能作为真实物理稳定或训练收敛证明**。

实施提交说明均包含“修改内容、修改原因、核验结果、未完成事项”。首个核心实现commit为 `a099200`；评估及README完成记录为I002，其准确hash由 `git log feature/loaded-4d` 查询。完成后按用户授权普通push实现分支，不覆盖原main对照。
