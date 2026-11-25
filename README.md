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

### Stage 2 VXH：本轮独立实验（实现与实际结果持续记录）

入口 `exp:g1-loaded-4d-stage2-vxh` 只限制 Gen：50% 前进 `vx=0.1–0.8,h=0.73`、
40% 原地高度 `h=0.60–0.80`、10% 持箱站立。Track/Gen 仍为 50/50，Track 使用
原 reference 四维指令和 tracking reward。两个零轴是速度约束，不是位置／朝向闭环。
训练与手动输入统一用显式限速 `[0.3,0.3,0.6,0.06]`，单位为对应指令单位/s；
vy/yaw command 在 Gen reset 时直接归零，不改实际物理速度。到达目标后再保持 3–5 s。
前进和高度组合未验收；手动输入不会被按高度静默缩小。

新 Gen 任务奖励为独立 vx（sigma=.10，weight=2）、pelvis height（.05，1）、
零 vy（.10，.3）、零 heading yaw-rate（.30，.3）；另加支撑及实际手箱几何软成本
（各 weight=-.5），保留动作正则和 3.5/s Gen tracking 常数。支撑只按 applied vx
近零触发，先留 .5 s 落脚宽限，再用 .5 s 平滑启用。手箱指标没有过滤接触传感器，
不能解释为已证实双手摩擦承重。训练前标定及冻结验收阈值见
[本轮标定](docs/loaded_4d_vxh/calibration_summary.json) 和
[验收标准](docs/loaded_4d_vxh/criteria.json)。旧 Stage2 preset 和报告保留。

从 Stage1 首次迁移（下面直接训练 1000 updates；已开始的实验请用续训命令）：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
STAGE1=/home/han/pyx/HMI_clean_runs/loaded_4d/stage1/WholeBodyTracking/20261124_105123-g1_loaded_4d_stage1_joint-locomotion/model_06000.pt
bash scripts_run/train_loaded_4d_vxh.sh transfer "$STAGE1" 1000 short_new 8192
```

脚本实际使用 `--algo.config.load-optimizer=False` 和
`--algo.config.reset-policy-std-on-resume=False`，恢复网络、normalizers、learned std，
新 optimizer 的 Actor/Critic 学习率均为 `1e-4`。脚本拒绝覆盖已有分段目录。
若直接运行原训练入口，追加同样两个选项及
`--logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh`。

继续同一 VXH 实验（把 `CHECKPOINT` 设为本轮结果记录中的真实路径）：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
CHECKPOINT=/absolute/path/to/stage2_vxh/model.pt
bash scripts_run/train_loaded_4d_vxh.sh resume "$CHECKPOINT" 1500 continuation_new 8192
```

`resume` 恢复 optimizer、std 和 normalizers。`num-learning-iterations` 是本段追加
updates；原 PPO 会从保存的显示 iteration 再开始编号。按每段 `segment.json` 中的
`completed_updates` 累加本轮 15000 预算，不能按最大编号减 6000；smoke 若继续使用也计入预算。
每段输出 `checkpoint.txt`，保存其最终 checkpoint；这只是恢复入口，不是“最佳模型”判断。

六个独立目标评估（每目标 10 trial，mean action，固定 locomotion frame 0、seed 0–9）：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
CHECKPOINT=/absolute/path/to/stage2_vxh/model.pt
python scripts_run/evaluate_loaded_4d.py --checkpoint "$CHECKPOINT" \
  --mode gen --suite vxh-targets --motion-id 1 --start-frames 0 \
  --seconds 15 --trials 10 --seed 0 --criteria docs/loaded_4d_vxh/criteria.json \
  --output /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/eval_targets.json
```

单独目标可用 `--sequence /path/one_target.json` 替代 `--suite vxh-targets`，JSON 例如
`[[0.1,0,0,0.73]]`。`--vxh-protocol` 仅用于以 Stage1 权重作新 command 协议的诊断对照。
`task_success` 检查到达目标后的响应、支撑、持箱和无失败；`horizon_complete` 仅说明存活。
全部控制步用于验收，10 Hz trace 保存 Target/Applied/Measured、实际位移、脚力／高度／滑移、
box-local wrist geometry、箱体姿态和 clearance。未执行的后续目标不计成功。

连续控制与 Track 保持分别执行：

```bash
python scripts_run/evaluate_loaded_4d.py --checkpoint "$CHECKPOINT" \
  --mode gen --suite vxh-sequence --motion-id 1 --seconds 70 --trials 10 \
  --criteria docs/loaded_4d_vxh/criteria.json \
  --output /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/eval_sequence.json
python scripts_run/evaluate_loaded_4d.py --checkpoint "$CHECKPOINT" \
  --mode track --trials 10 --start-frames 0 --seed 0 \
  --output /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/eval_track.json
```

连续序列是站立→0.1→0.4→停止→h=.60→h=.80→再次前进，中途不 reset。
新 checkpoint 的 Viser 命令如下，网页为 http://localhost:8012；只有 vx 和 height 可编辑，
另外两轴显示固定零，预设包含三个速度和三个高度。控件显示训练范围，不代表范围已全部通过。

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
CHECKPOINT=/absolute/path/to/stage2_vxh/model.pt
python src/holosoma/holosoma/eval_agent.py --checkpoint="$CHECKPOINT" \
  --viser.config.enabled --viser.config.loaded-4d-controls --viser.config.no-show-reference \
  --viser.config.port=8012 --viser.config.num-envs=1 --training.num-envs=1 \
  --training.headless=True --training.max-eval-steps=30000 --training.export-onnx=False \
  --simulator.config.sim.max-episode-length-s=600.0 \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/viewer \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.0 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.enabled=True \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.motion-id=1 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.bin-id=None \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.local-frame=0 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.randomize-within-bin=False
python -m tensorboard.main --logdir=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh \
  --host=127.0.0.1 --port=6020 --load_fast=false
```

TensorBoard 单独终端运行，访问 http://localhost:6020。重点检查
`Env/loaded_4d/gen_reward_rate/*`、`Env/loaded_4d/sampled/*` 与逐目标评估，
不要仅凭总分或平均存活时间选 checkpoint。实际训练、结果和未通过项见
[本轮结果记录](HMI_LOADED_4D_STAGE2_VXH_RESULTS.md)。

本轮正在使用的短训目录（`model.pt` 会每 100 个显示 iteration 更新）：

```bash
RUN=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/short_1000/WholeBodyTracking/20261125_145544-g1_loaded_4d_stage2_vxh-locomotion
CHECKPOINT="$RUN/model.pt"
tail -f /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/short_1000.log
```

本轮已启动 `screen` 会话 `hmi-vxh-round`，等待短训结束后顺序执行
`scripts_run/continue_loaded_4d_vxh.sh`：每 1500 updates 做六目标和三 Track 各2-trial筛查，
出现稳态响应后才测连续切换；累计不超过15000 updates。三个连续窗口没有前进/高度响应，
或 Track 出现严重退化时保存结果并停止追加训练。结束时对筛选候选做每目标10trial、三Track各10trial，
有响应才做连续10trial；脚本不放宽验收阈值、不扩大范围。筛选同时考虑任务成功、误差、存活、
Track完成率及相对Stage1的body/object误差，不能把 `best_checkpoint.txt` 自动解释为可部署模型。

```bash
screen -ls
cat /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/progress.json
cat /home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/round_driver.log
```

当前运行期间无需重复启动。中断或诊断后，以真实保存checkpoint和该checkpoint对应的累计updates恢复：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh
RUN=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh/short_1000/WholeBodyTracking/20261125_145544-g1_loaded_4d_stage2_vxh-locomotion
CHECKPOINT="$RUN/model.pt"
COMPLETED=$(python scripts_run/summarize_loaded_4d_vxh.py --count-updates "$CHECKPOINT")
bash scripts_run/continue_loaded_4d_vxh.sh "$CHECKPOINT" "$COMPLETED"
```

已进入后续分段时，把RUN改成最近已完成段的目录，或从该段 `checkpoint.txt` 读取路径。
计数工具沿保存checkpoint的分段祖先累计updates，正确处理重复显示编号；不清空optimizer。
已有同名分段会拒绝覆盖，保留此前失败和完成产物。中途失败应使用新的分段名继续，
具体段命令为 `train_loaded_4d_vxh.sh resume CHECKPOINT UPDATES NEW_SEGMENT 8192`。

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

### Stage 2：在 Viser 网页中固定或切换四维 command

**当前模型状态（2026-11-25）：** 用户已确认下面的 `model_20999.pt` 在 Track viewer 中仍能跟踪 reference，
但已测试的 Gen 指令失败，包括训练范围内的 `vx=0.1`（单腿站立、没有明显前进）和下蹲
（单腿支撑、身体变化很小）。下面是查看与诊断入口，不代表该模型已通过四维控制验收。
完整的训练日志、reward、command 通路及 Track/Gen 对比见
[Stage 2 Track / Gen 分析报告](HMI_LOADED_4D_STAGE2_TRACK_GEN_ANALYSIS.md)。

使用训练好的 Stage 2 policy，将评估环境设为 **Gen**，并启用
`--viser.config.loaded-4d-controls`。网页可以选择预设或手动调整
`[vx, vy, yaw_rate, pelvis_height]`，无需重新训练。

下面使用当前 Stage 2 的 `model_20999.pt`，从行走 reference 的 frame 0 初始化持箱状态。
`MOTION_ID` 在这里只决定初始状态，后续动作由四维 command 和 policy 决定；任意新 command
没有对应的逐帧 reference，Gen viewer 会隐藏蓝色 reference 和旧 object endpoint goal。

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

RUN_DIR=/home/han/pyx/HMI_clean_runs/loaded_4d/stage2/WholeBodyTracking/20261124_165511-g1_loaded_4d_stage2_joint-locomotion
CHECKPOINT="$RUN_DIR/model_20999.pt"
MOTION_ID=1

python src/holosoma/holosoma/eval_agent.py \
  --checkpoint="$CHECKPOINT" \
  --viser.config.enabled \
  --viser.config.loaded-4d-controls \
  --viser.config.no-show-reference \
  --viser.config.port=8012 \
  --viser.config.num-envs=1 \
  --training.num-envs=1 \
  --training.headless=True \
  --training.max-eval-steps=30000 \
  --training.export-onnx=False \
  --simulator.config.sim.max-episode-length-s=600.0 \
  --logger.base-dir=/home/han/pyx/HMI_clean_runs/loaded_4d/eval_viewer \
  --command.setup-terms.motion-command.params.motion-config.hybrid.track-ratio=0.0 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.enabled=True \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.motion-id="$MOTION_ID" \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.bin-id=None \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.local-frame=0 \
  --command.setup-terms.motion-command.params.motion-config.eval-fixed-reset.randomize-within-bin=False
```

打开 **http://localhost:8012**。启动时保持站立目标 `[0, 0, 0, 0.73]`：

1. 在 **Command preset** 选择动作，或直接调整四个滑块/数值。
2. 点击 **Apply & hold command**，开始执行并保持这组目标；只选预设或编辑数值时不会发送。
3. 想切换时再次编辑并点击同一按钮。切换本身不重置机器人或箱体，沿用现有 command 限速。
4. **Resume random commands** 恢复训练配置的随机采样。

| 预设 | `[vx, vy, yaw_rate, pelvis_height]` |
|---|---|
| Stand / rise | `[0, 0, 0, 0.73]` |
| Forward 0.1 / 0.4 / 0.8 m/s | `[0.1 / 0.4 / 0.8, 0, 0, 0.73]`，下拉框中为三个独立选项 |
| Backward | `[-0.1, 0, 0, 0.73]` |
| Left / Right | `[0, ±0.1, 0, 0.73]` |
| Turn left / right | `[0, 0, ±0.2, 0.73]` |
| Squat | `[0, 0, 0, 0.66]` |

**Command response** 显示目标值 `Target`、该控制步实际使用的指令 `Applied`、实际测量值
`Measured`，以及 episode 步数和 reset 次数。目标变化时 `Applied` 会逐步接近 `Target`。
跌倒等失败仍会触发原有 reset；手动目标在 reset 后继续保持，reset 计数可以识别这些重新开始。
此命令将 episode 上限设为 600 s；不会因 command 更新或 reference clip 结束而 reset。

当前外部输入范围为 `vx±0.8`、`vy±0.4`、`yaw_rate±1.0`、height `0.60–0.78`。
但这个 Stage 2 run 的 Gen 训练是 `axes=single`，采样范围为 `vx±0.15`、`vy±0.10`、
`yaw_rate±0.20`、height `0.66–0.73`。因此用户指定的 `0.4/0.8 m/s` 是训练采样范围外测试，
手动组合多个轴也需实际检验；接口接受某个值不表示 policy 已可靠掌握。
上述滑块仅控制当前可视化的 Gen envs，默认关闭；原 Stage 1 Track viewer 操作保持不变。

### Stage 2：连续 command 量化评估

评估输出中的 `completed_trials` 仅表示运行至 horizon，尚未按 command 误差或支撑姿态判定成功。
诊断当前模型时应先逐个测范围内目标，再测连续组合；同时检查 `trace` 的 Target/Applied/Measured、
稳态误差与 reset，避免前一个目标失败后误判后续目标已被测试。

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
