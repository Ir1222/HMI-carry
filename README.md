# Hybrid Motion Imitation

<table>
  <tr>
    <td align="center" width="50%">
      <strong>Box Moving</strong><br>
      <img src="docs/assets/readme/move_box_1.gif" alt="Unitree G1 box-moving task" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Box Climbing</strong><br>
      <img src="docs/assets/readme/climb_box_1.gif" alt="Unitree G1 box-climbing task" width="100%">
    </td>
  </tr>
</table>

This is an unofficial implementation and extension of
[GfR](https://jiashunwang.github.io/GfR/) and
[HIL](https://jiashunwang.github.io/HIL/), built on top of
[Holosoma](https://github.com/amazon-far/holosoma). It is not the official
code release for either paper.

This release provides two tasks for the
29-DoF Unitree G1:

```text
exp:g1-29dof-wbt-hybrid-climb
exp:g1-29dof-wbt-hybrid-object
```

The same hybrid learning framework is used for both box climbing and box
moving. Its key ideas are:

1. **The actor does not observe the target reference motion.** We find that this
   design can achieve high-quality motion tracking across a wide range of
   tasks; see the [HIL paper](https://jiashunwang.github.io/HIL/) for more details.
   The actor only observes the robot state and sparse task conditions or goals,
   while the reference motion is used to define the goals and tracking rewards.
   Because the policy is conditioned on task goals rather than a reference
   motion, the learned behavior can easily transfer to new task conditions.

2. **Training combines two types of environments.**
   - In **reference-tracking environments**, goals and task conditions are extracted from the
     reference motion and the policy is trained with tracking rewards.
   - In **general RL environments**, goals and task conditions are randomized
     and the policy is trained with sparse goal-reaching rewards.

3. **One policy is jointly optimized on two types of environments.** The actor always uses the
   same observation interface across track and general RL envs, while the critic receives task-specific privileged information. This allows behavior learned from a small reference set to generalize to a wide range of task conditions and goals.

## Setup

Run from the repository root:

```bash
bash scripts/setup_isaacsim.sh
source scripts/source_isaacsim_setup.sh
wandb login
```

## Loaded-4D：当前实现与训练指令

代码工作目录为 `/home/han/pyx/HMI_clean`，实现分支为 `feature/loaded-4d`。
本任务从已经持箱的 reference state 开始，箱体是自由动态刚体；外部 command 为
`[vx, vy, yaw_rate, pelvis_height]`，分别表示 pelvis heading frame 的前后/侧向速度
（m/s）、heading 变化率（rad/s）、pelvis 离地高度（m）。

Stage 1 **可以从头训练，不需要任何已有 checkpoint**：三个 Stage 1 preset 默认
`checkpoint=None`、Track ratio=1.0，使用原 PPO 随机初始化网络、空 optimizer state、
新 observation normalizers 和 `init_noise_std=1.0`。不会自动寻找旧日志或旧模型恢复。
下面显式写出 `--training.checkpoint=None`。三条 reference 联合训练是默认主实验；
locomotion/squat 是同一实现的诊断配置。原版 Object/Climb 的命令保留在后面的章节中。

2026-11-25，用户已确认：当前 joint Stage 1 checkpoint 可以通过 Viser，分别以
`MOTION_ID=0`、`1`、`2` 查看三个 reference 下的 policy 表现。下面保存这套实际使用的方法。
这项确认说明逐 motion 查看流程可用；训练收敛和量化完成率仍以评估结果为准。
以下训练和评估命令会在执行时启动仿真；CPU 数据检查不会启动仿真。
详细设计、修改记录及验证边界见 [implementation plan](HMI_LOADED_4D_IMPLEMENTATION_PLAN.md)。

### 每次新开终端：激活本机环境

每次打开终端进行训练、Viser 查看或量化评估，都先执行下面两行。
训练和评估使用同一个隔离 runtime；评估放在新终端运行，训练终端可以继续运行：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
```

终端环境前缀应为 `(/home/han/pyx/HMI_clean_runtime/env)`，`which python` 应指向
`/home/han/pyx/HMI_clean_runtime/env/bin/python`。激活脚本同时设置当前源码的
`PYTHONPATH` 和 `CUDA_VISIBLE_DEVICES=0`。本机已有依赖，无需每次执行上面的安装流程。
下方主要训练、查看和评估命令也包含这两行，可直接复制到新终端执行。

首次使用或调整数据路径后，可在该环境做一次 CPU 数据检查：

```bash
python scripts_run/evaluate_loaded_4d.py --validate-data
```

`datasets/loaded_4d/joint/` 只包含三条 reference 的相对 symlink，`squat/` 只包含两条
蹲起。原始 NPZ 位于同级旧工作区的
`g1_carry_baseline_new/HMI/datasets/unified_hoi_v1/motions/`，不会被训练代码修改。
新机器需要自行提供这些原数据；仓库不包含 NPZ 副本。移动数据位置时，将
`--command.setup-terms.motion-command.params.motion-config.motion-dir`
指向只含相应文件的目录；locomotion-only 使用 `motion-file`。运行时会核对文件集合、
SHA256、帧数和 50 Hz，不接受加载器静默跳过某个 clip。

### Stage 1：三条 reference 联合、从头训练

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

python src/holosoma/holosoma/train_agent.py \
  exp:g1-loaded-4d-stage1 \
  logger:disabled \
  --training.checkpoint=None \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/stage1
```

`logger:disabled` 关闭 W&B，仍保存 TensorBoard、配置和 checkpoint。实际输出目录以
启动日志为准。默认 200 Hz physics、decimation=4、50 Hz policy、15 s episode，
Actor/Critic 输入为 113D/315D；保持原 MLP、joint-position action、PD 和 PPO。
不应使用原版 111D/316D 或旧 carry 357D/207D checkpoint 初始化该任务。

建议先执行同一 preset 的小规模启动检查；它也从头初始化，不等于正式训练已通过：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

python src/holosoma/holosoma/train_agent.py \
  exp:g1-loaded-4d-stage1 \
  logger:disabled \
  --training.checkpoint=None \
  --training.num-envs=64 \
  --algo.config.num-learning-iterations=3 \
  --algo.config.save-interval=1 \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/smoke
```

单技能短训练用于定位行走与蹲起问题，两条命令分别执行：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

python src/holosoma/holosoma/train_agent.py \
  exp:g1-loaded-4d-stage1-locomotion \
  logger:disabled \
  --training.checkpoint=None \
  --training.num-envs=128 \
  --algo.config.num-learning-iterations=200 \
  --algo.config.save-interval=100 \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/locomotion

python src/holosoma/holosoma/train_agent.py \
  exp:g1-loaded-4d-stage1-squat \
  logger:disabled \
  --training.checkpoint=None \
  --training.num-envs=128 \
  --algo.config.num-learning-iterations=200 \
  --algo.config.save-interval=100 \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/squat
```

这些短训练不作为收敛验收。单技能 checkpoint 与 joint 的数据集合不同，当前 contract
会拒绝直接混用；主实验应使用 joint preset 从头训练。

### Stage 2：从新的 joint Stage 1 checkpoint 迁移

将示例路径替换为**本实现**训练并验收后的 joint Stage 1 checkpoint：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

python src/holosoma/holosoma/train_agent.py \
  exp:g1-loaded-4d-stage2 \
  logger:disabled \
  --training.checkpoint=/absolute/path/to/loaded_4d_stage1/model.pt \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --algo.config.reset-policy-std-on-resume=False \
  --command.setup-terms.motion-command.params.loaded-4d.axes=single \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2
```

Stage 2 默认重新建立 50/50 Track/Gen 分组，恢复 Actor、Critic、两个 normalizer、
optimizer 和 learned std。`num-learning-iterations` 表示本次**追加** updates。
先以 `axes=single` 学习单轴；验证后从 Stage 2 checkpoint 继续，将同一个参数改为
`pair`，再改为 `combined`。可用 `loaded-4d.dwell-s='[3.0, 5.0]'` 调整切换周期。
初始采样范围为 vx±0.15、vy±0.10、yaw±0.20、height 0.66–0.73；扩大范围须单独验收。

### Stage 1：用 Viser 逐条查看三个 reference（已验证）

方法是：先定义 `RUN_DIR` 和 `CHECKPOINT`，再定义一个 `MOTION_ID`，最后执行同一条
完整 viewer 命令。三个 ID 使用同一个 joint policy，分别从对应 reference 的 frame 0 初始化。

| `MOTION_ID` | Reference |
|---|---|
| `0` | `continuous_grasped_squat_stand.npz` |
| `1` | `loaded_locomotion.npz` |
| `2` | `repeated_grasped_squat_stand.npz` |

下面是当前 run 的可直接执行命令，默认查看行走。以后训练新 run 时，只需更新 `RUN_DIR`
以及要查看的 checkpoint 文件名。`RUN_DIR` 末尾的 `-locomotion` 是运行目录标签，
这个 run 的实际配置和日志确认加载了三条 reference：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

RUN_DIR=/home/han/pyx/HMI_clean_runs/loaded_4d/stage1/WholeBodyTracking/20261124_105123-g1_loaded_4d_stage1_joint-locomotion
CHECKPOINT="$RUN_DIR/model_06000.pt"
MOTION_ID=1

python src/holosoma/holosoma/eval_agent.py \
  --checkpoint="$CHECKPOINT" \
  --viser.config.enabled \
  --viser.config.port=8012 \
  --viser.config.num-envs=1 \
  --training.num-envs=1 \
  --training.headless=True \
  --training.max-eval-steps=30000 \
  --training.export-onnx=False \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/eval_viewer \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.enabled=True \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.motion-id="$MOTION_ID" \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.bin-id=None \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.local-frame=0 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.randomize-within-bin=False
```

启动后，在本机浏览器打开 **http://localhost:8012**。蓝色机器人为 reference，
另一台机器人及其自由箱体展示 policy 的实际物理表现。

切换动作时，在**评估终端**按 `Ctrl+C` 结束当前 viewer，执行 `MOTION_ID=0`、
`MOTION_ID=1` 或 `MOTION_ID=2` 中的一条，再执行上面从 `python` 开始的完整命令。
每次选择一个整数；该变量在进程启动时读取，不能通过修改另一个终端的变量切换已运行的 viewer。
保留 checkpoint 的 joint 数据集合，不需要改 `motion-dir` 或换成单技能 preset。

`RUN_DIR`、`CHECKPOINT`、`MOTION_ID` 都只保存在当前 shell；新开终端时要重新定义。
要查看训练最近保存的权重，可设 `CHECKPOINT="$RUN_DIR/model.pt"`，再启动 viewer。
已运行的 viewer 不会自动加载新权重；比较训练进度时优先选择 `model_06000.pt` 这类编号文件。
失败或 clip 结束都会 reset，完整动作完成率应通过下方量化评估确认。

若曾遇到 `[Viser] Not available (No module named 'websockets.asyncio')`，本机 Isaac Sim
自带 `websockets 12.0` 会遮蔽已安装的 `15.0.1`。当前 `eval_agent.py` 在启用 Viser 时，
会在启动 simulator 前导入 Viser，避免该冲突；适用于原版和 Loaded-4D checkpoint。
更新代码后重新启动评估进程即可，正在运行的训练不需要重启，也不需要为此升级依赖。

### Stage 1：三条 reference 的量化评估

该命令无窗口运行，逐条 reference 从 frame 0 开始，每条 10 个 trial。可在独立终端直接执行：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

RUN_DIR=/home/han/pyx/HMI_clean_runs/loaded_4d/stage1/WholeBodyTracking/20261124_105123-g1_loaded_4d_stage1_joint-locomotion
CHECKPOINT="$RUN_DIR/model_06000.pt"

python scripts_run/evaluate_loaded_4d.py \
  --checkpoint "$CHECKPOINT" \
  --mode track \
  --trials 10 \
  --start-frames 0 \
  --output /home/han/pyx/HMI_clean_runs/loaded_4d/eval_stage1/model_06000_all_refs.json
```

不指定 `--motion-id` 就评估全部三条 reference；这里不会读取 shell 中的 `MOTION_ID`。
脚本逐条打印 motion、起始帧、trial 编号、结果和 termination terms；共 30 个 trial，
全部结束并写入 JSON 后打印 `Saved 30 trials to ...`。仅看到
`Shutting down simulation app` 不代表评估完成。若需留存终端日志，可在命令末尾加
`2>&1 | tee /tmp/loaded_4d_eval.log`。更换 checkpoint 时同步更改输出 JSON 文件名。

可加 `--start-frames 0 25 50` 检查多个 RSI 起点，或 `--motion-id 0` 选择一条 clip。
`--stochastic` 使用 learned exploration std；默认使用 policy mean，并保持训练时的
observation noise 配置和冻结的 normalizers。当前脚本在关闭 Isaac 之前打印原始异常；
多 trial 使用 `torch.no_grad()`，避免跨 trial reset 的 inference tensor 修改错误。

### Stage 2：连续 command 评估

60 s 连续 Gen 指令评估，包含站立、前后/侧向、转弯、蹲起及组合：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

python scripts_run/evaluate_loaded_4d.py \
  --checkpoint /absolute/path/to/loaded_4d_stage2/model.pt \
  --mode gen --seconds 60 --trials 10 \
  --output /home/han/pyx/HMI_clean_runs/loaded_4d/eval_stage2/results.json
```

`--sequence /path/commands.json` 可提供四维 command 的 JSON 数组，按等时段执行；
`--viewer` 开启本地 simulator viewer。工具记录 command 误差、robot/object tracking
误差、box clearance、reset、首次失败原因和响应序列；不会将失败重置后的片段拼成成功。
`completion_fraction` 只表示完成率，command 误差和持箱情况须同时验收。

已有环境的外部输入接口：

```python
command = env.command_manager.get_state("motion_command")
command.set_command([0.10, 0.00, 0.15, 0.70])  # 所有 Gen envs；可显式传 env_ids
command.release_external_command()  # 回到随机采样；不重置物理状态
```

调用 setter 只改变目标；policy 实际接收的指令按控制周期插值。可分别读取
`target_command`、`task_command`、`applied_command`；Track envs 拒绝外部覆盖。

## Training

Training uses two stages:

1. Stage 1 trains track environments only.
2. Stage 2 resumes Stage 1 and trains a 50/50 mix of track and gen
   environments.

### Climb Stage 1

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-climb \
  logger:wandb \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --simulator.config.scene.env-spacing=5.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

### Climb Stage 2

Replace `CLIMB_STAGE1_CHECKPOINT` with the Stage 1 `model.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-climb \
  logger:wandb \
  --training.checkpoint=CLIMB_STAGE1_CHECKPOINT \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --algo.config.reset-policy-std-on-resume=True \
  --algo.config.init-noise-std=0.45 \
  --simulator.config.scene.env-spacing=5.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.5 \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-start-at-timestep-zero-prob=0.2 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-std-xyz='[0.3, 0.3, 0.01]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-clip-xyz='[1.0, 1.0, 0.02]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-std='[0.05, 0.05, 0.3]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-clip='[0.15, 0.15, 1.0]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

### Object Stage 1

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-object \
  logger:wandb \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --simulator.config.scene.env-spacing=4.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

### Object Stage 2

Replace `OBJECT_STAGE1_CHECKPOINT` with the Stage 1 `model.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-hybrid-object \
  logger:wandb \
  --training.checkpoint=OBJECT_STAGE1_CHECKPOINT \
  --training.num-envs=8192 \
  --algo.config.num-learning-iterations=15000 \
  --algo.config.reset-policy-std-on-resume=True \
  --algo.config.init-noise-std=0.45 \
  --simulator.config.scene.env-spacing=4.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.5 \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.05 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-start-at-timestep-zero-prob=0.2 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-std-xyz='[0.5, 0.5, 0.001]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-clip-xyz='[1.0, 1.0, 0.001]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-std='[0.001, 0.001, 0.4]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-clip='[0.001, 0.001, 0.8]' \
  --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.enabled=True \
  --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.initial-scale=0.3 \
  --termination.terms.bad-tracking.params.gen-bad-object-pos-z-threshold=0.6 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

The same commands are available in
[`scripts_run/run_train.sh`](scripts_run/run_train.sh).


## Evaluation

We use Viser for web-based visualization. Training and evaluation can run
headlessly on a remote server, while the simulation is viewed from any machine
with a web browser—no local workstation or Isaac Sim window is required. The
examples serve Viser at `http://localhost:8012`.

### Climb Stage 1: track

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=CLIMB_STAGE1_CHECKPOINT \
  --viser.config.enabled \
  --viser.config.port=8012 \
  --training.num-envs=2 \
  --training.headless=True \
  --training.max-eval-steps=10000 \
  --training.export-onnx=False \
  --simulator.config.scene.env-spacing=5.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file='' \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
  --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
  --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
  --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.body-proximity.is-timeout=True \
  --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.bad-tracking.is-timeout=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

### Climb Stage 2: gen

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=CLIMB_STAGE2_CHECKPOINT \
  --viser.config.enabled \
  --viser.config.port=8012 \
  --viser.config.no-show-reference \
  --training.num-envs=2 \
  --training.headless=True \
  --training.max-eval-steps=10000 \
  --training.export-onnx=False \
  --simulator.config.scene.env-spacing=0.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file='' \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/climbox1' \
  --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
  --command.setup-terms.motion-command.params.motion-config.goal-orientation.robot-yaw-only=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-std-xyz='[0.3, 0.3, 0.01]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.pos-clip-xyz='[1.0, 1.0, 0.02]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-std='[0.05, 0.05, 0.3]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.gen-step-zero-root-noise.rpy-clip='[0.15, 0.15, 1.0]' \
  --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.body-proximity.is-timeout=True \
  --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.bad-tracking.is-timeout=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

### Object Stage 1: track

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=OBJECT_STAGE1_CHECKPOINT \
  --viser.config.enabled \
  --viser.config.port=8012 \
  --training.num-envs=2 \
  --training.headless=True \
  --training.max-eval-steps=10000 \
  --training.export-onnx=False \
  --simulator.config.scene.env-spacing=0.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=1.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file='' \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
  --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False \
  --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.bad-tracking.is-timeout=True \
  --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.body-proximity.is-timeout=True
```

### Object Stage 2: gen

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=OBJECT_STAGE2_CHECKPOINT \
  --viser.config.enabled \
  --viser.config.port=8012 \
  --viser.config.no-show-reference \
  --training.num-envs=2 \
  --training.headless=True \
  --training.max-eval-steps=10000 \
  --training.export-onnx=False \
  --simulator.config.scene.env-spacing=0.0 \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file='' \
  --command.setup-terms.motion-command.params.motion-config.motion-dir='holosoma/data/motions/g1_29dof/whole_body_tracking/movebox' \
  --command.setup-terms.motion-command.params.motion-config.eval-sample-random-timestep=False \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-std-xyz='[0.5, 0.5, 0.001]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.pos-clip-xyz='[1.0, 1.0, 0.001]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-std='[0.001, 0.001, 0.4]' \
  --command.setup-terms.motion-command.params.motion-config.hybrid.object-goal-noise.rpy-clip='[0.001, 0.001, 0.8]' \
  --curriculum.setup-terms.hybrid-gen-noise-curriculum.params.initial-scale=1.0 \
  --termination.terms.body-proximity.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.body-proximity.is-timeout=True \
  --termination.terms.bad-tracking.func='holosoma.managers.termination.terms.common:timeout_exceeded' \
  --termination.terms.bad-tracking.is-timeout=True \
  --command.setup-terms.motion-command.params.motion-config.hybrid.size-randomization.enabled=False
```

The same commands are available in
[`scripts_run/run_play.sh`](scripts_run/run_play.sh).

Important evaluation behavior:

- `training.num-envs=2` controls the simulated environment count. Viser can
  display up to 16 by default, so no separate Viser count is needed.
- `eval-sample-random-timestep=False` starts every reset at timestep zero.
  Task reset noise still applies.
- The play commands disable early failure termination for longer sessions.
- `no-show-reference` hides the displaced blue reference in gen mode.

## Hybrid method

See [docs/wbt-hybrid.md](docs/wbt-hybrid.md) for the track/gen design,
observations, rewards, reset and goal noise, curriculum, termination
conditions, and checkpoint compatibility.

## Citation

This implementation is based on the learning frameworks introduced in GfR
(RSS 2026) and HIL (ACM TOG 2026). If it is useful for your research, please
cite the two papers:

```bibtex
@inproceedings{wang2026generalizing,
  title={Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework},
  author={Wang, Jiashun and Mungai, M. Eva and Li, He and Sleiman, Jean Pierre and Hodgins, Jessica and Farshidian, Farbod},
  booktitle={Robotics: Science and Systems},
  year={2026}
}

@article{wang2026hil,
  title={{HIL:} Hybrid Imitation Learning for Dynamic Athletic Control},
  author={Wang, Jiashun and Jiang, Yifeng and Zhang, Haotian and Tessler, Chen and Rempe, Davis and Hodgins, Jessica and Peng, Xue Bin},
  journal={{ACM} Transactions on Graphics},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

This project is built on [Holosoma](https://github.com/amazon-far/holosoma).
