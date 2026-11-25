# Loaded-4D Stage 2：Track / Gen 行为与四维指令审查

审查日期：2026-11-25。审查时的代码基线：`feature/loaded-4d`，`0dbbbce`；此处不是声称训练时使用了这个 commit。本报告针对下面的 **`model_20999.pt`**，不代表所有 Stage 2 实验。

```text
/home/han/pyx/HMI_clean_runs/loaded_4d/stage2/WholeBodyTracking/
  20261124_165511-g1_loaded_4d_stage2_joint-locomotion/model_20999.pt
```

## 1. 结论与证据边界

**当前 checkpoint 保留了用户在 viewer 中观察到的 reference tracking 能力，但已测试的 Gen 指令未达到目标控制要求。`vx=0.1` 属于 Gen 训练范围，仍出现单腿站立且没有明显前进；下蹲同样出现单腿支撑、身体变化很小。因此，不能用 `0.4/0.8 m/s` 超出训练范围解释主要失败，也不能继续把该模型称为已经成功的四维 goal-conditioned controller。**

最有依据的解释是：当前 Gen 的优化目标允许保持姿态、存活而少响应指令；同时没有约束站立/下蹲的支撑方式。Track 的逐帧 body/object tracking 会限制动作形态，但这些限制在 Gen 变成常数，不能保证同一个网络在独立目标下延续相同动作。这个解释有明确代码和奖励数值支持，尚未通过干预实验确认它是唯一或全部原因。

本次完成的是源码审查、CPU checkpoint/NPZ 检查、现有 TensorBoard 与运行配置分析。没有启动训练、Isaac Sim 或 GPU rollout，没有修改模型、数据、训练配置或 Python 源码。

| 证据类别 | 本报告可下的结论 | 不能据此声称的结论 |
|---|---|---|
| 用户实际 viewer 反馈 | 同一个 Stage 2 checkpoint 的 Track 可追踪；已测试 Gen 指令失败；`vx=0.1`、下蹲的单腿现象明确 | 精确速度、下蹲高度、持续时间、每个目标的成功率 |
| 保存的配置、checkpoint、源代码 | 实际训练范围、reward、输入、mask、时序和终止语义 | 仅看权重就知道闭环轨迹或足底接触力 |
| 完整 TensorBoard 标量 | 训练期间分组平均误差、episode 长度、回报、采样变化 | 某个固定指令的稳态响应、某次网页操作是否已到达 Applied |
| 本报告解析计算 | 某种假设状态在当前 reward 下能获得多少分 | 模型真实轨迹恰好等于该假设状态，或算法必然收敛到它 |

审查时，`HMI_clean_runs/loaded_4d/{eval_stage1,eval_stage2,eval_viewer}` 下未找到保存的 JSON/CSV/NPZ 量化轨迹。viewer 的 YAML 保留启动参数，不保存每次点击目标或连续 Measured 曲线。本报告没有把缺失的测量值补成实验结果。

## 2. 被审查的模型和训练过程

以下路径简称用于后文定位证据，均为本地实际路径：

| 简称 | 绝对路径 |
|---|---|
| `REPO` | `/home/han/pyx/HMI_clean` |
| `SRC` | `/home/han/pyx/HMI_clean/src/holosoma/holosoma` |
| `RUN` | `/home/han/pyx/HMI_clean_runs/loaded_4d/stage2/WholeBodyTracking/20261124_165511-g1_loaded_4d_stage2_joint-locomotion` |
| `RUNTIME` | `/home/han/pyx/HMI_clean_runtime` |

### 2.1 实际配置

配置依据为 `RUN/holosoma_config.yaml` 和 checkpoint 的 `experiment_config`，而非仅按当前 preset 推测。两者递归比较只有 Actor `output_dim` 从占位符 `robot_action_dim` 解析成 `29` 的差异。

| 项目 | 本次 run 实际值 |
|---|---|
| 环境 | `Loaded4DTrackingManager`，8192 envs，seed 42 |
| 初始化模型 | Stage 1 joint run `20261124_105123-g1_loaded_4d_stage1_joint-locomotion/model_06000.pt` |
| 更新区间 | TensorBoard iteration 6000–20999，15000 个 PPO iteration |
| Track / Gen | 固定分组，0.5 / 0.5 |
| reference | joint 目录中的三条 motion；run 名末尾 `-locomotion` 不代表只加载行走 |
| Gen 目标配置 | `axes=single`，`stand_probability=0.2`，dwell 3–5 s |
| Gen 采样范围 | `vx∈[-0.15,0.15]`、`vy∈[-0.10,0.10]` m/s；`yaw_rate∈[-0.20,0.20]` rad/s；`height∈[0.66,0.73]` m |
| 目标过渡 | 随机前三维 0.8 s smoothstep；height 限速 0.06 m/s |
| episode | 15 s；Track 还会在 clip 末帧结束 |
| 控制频率 | physics 200 Hz / dt 0.005 s；decimation 4；policy 50 Hz / dt 0.02 s |
| Actor / Critic | 113D / 315D；512/256/128 ELU；history length 1，无 history augmentation |
| Action | 29D；原 WBT joint-position action、PD 和基于 effort/P-gain 的缩放 |
| PPO | rollout 24 steps、5 epochs、4 minibatches、clip 0.2、gamma 0.99、lambda 0.95、entropy coefficient 0.002、adaptive KL target 0.01；无 symmetry loss |
| 迁移 | Actor、Critic、两个 normalizer、optimizer 和 learned std 恢复；`reset_policy_std_on_resume=False` |
| Curriculum | 只有 `AverageEpisodeLengthTracker`；**没有自动把 single 扩展到 pair/combined，也没有扩大 command 范围的 curriculum** |

`RUN/train.log` 记录末尾保存 `model_20999.pt`、`model.pt` 后正常退出。编号 checkpoint 包含 06000、09000、12000、15000、18000、20999；这些文件存在表示训练产物可用，不表示任务验收成功。

模型 SHA256：`54802d217ac9cea92763ccb978381f0fee10498b2e8c23045e3601bb58beb63c`。

配置 SHA256：`12ecdaaee1e36f19086251d4544c89562bcc0ac89dc6a593ca7eb100ff355068`。

### 2.2 Reference 覆盖与 Gen 覆盖是两回事

数据视图为 `REPO/datasets/loaded_4d/joint`，指向旧仓库 `g1_carry_baseline_new/HMI/datasets/unified_hoi_v1/motions/` 的原始 NPZ。manifest、帧数及 command 提取见 `config_values/wbt/g1/loaded_4d.py:MOTIONS`、`scripts_run/evaluate_loaded_4d.py:validate_data`。

| Motion ID | 文件 | 帧数 / fps / 首末帧间隔 | pelvis 高度 | heading `vx` |
|---|---|---|---|---|
| 0 | `continuous_grasped_squat_stand.npz` | 351 / 50 / 7.00 s | 约 0.6069–0.7269 m | 约 ±0.0155 m/s |
| 1 | `loaded_locomotion.npz` | 110 / 50 / 2.18 s | 约 0.7231–0.7418 m | 约 0.2934–0.7871 m/s |
| 2 | `repeated_grasped_squat_stand.npz` | 651 / 50 / 13.00 s | 约 0.6069–0.7269 m | 约 ±0.0155 m/s |

行走 reference 同时包含 `vy≈[-0.0819,0.2379]`、`yaw_rate≈[-2.124,1.498]`，这些量随 reference 时间相关变化。三条数据并没有提供纯后退、纯横移、原地转向、低位前进的完整联合覆盖。

`[0.1,0,0,0.73]` **在 Gen 采样支持范围内**，但并非行走 reference 中某一段稳定的慢速前进轨迹。`0.4/0.8` 的标量速度接近部分 Track 数据，不等于完整四维目标、物理状态及目标保持时长都被训练覆盖。不能把各维度的极值组合当作能力范围。

## 3. Track 与 Gen 到底有哪些不同

| 项目 | Track 训练 / viewer | Gen 训练 | Gen 交互 viewer |
|---|---|---|---|
| Policy | 同一个 Actor、同一个 normalizer、同一 action 语义 | 相同 | 相同 checkpoint，使用 Actor mean action |
| 4D command | 每帧 reference 的 pelvis command | 独立随机采样并过渡 | 用户目标经外部限速后输入 |
| reference time | 每个 control step 前进一帧 | 停在 reset frame | 停在 reset frame；隐藏蓝色 reference |
| Robot / object tracking reward | 8 项逐帧跟踪误差 | 8 项改为常数，每项 raw 0.5 | 运行相同环境，但 reward 不会在推理时训练网络 |
| command reward | Gen mask 下为 0 | velocity/yaw/height 三项 | 相同评价公式 |
| Actor reference 输入 | 无完整 reference pose、无 phase、无 Track/Gen flag | 相同 | 相同 |
| Critic reference 输入 | hybrid reference terms 有效 | 相应 reference terms 清零 | Critic 不决定推理 action |
| reset 初态 | reference state，训练以 RSI 为主 | reference state | 用户固定 motion ID/frame 0 |
| 结束条件 | reference bad tracking、clip end、物理失败、timeout | 物理失败、脚距离过小、15 s timeout | 同样物理失败；用户将 timeout 改为 600 s |

将 `track-ratio` 改为 0 不会加载另一套“Gen policy”。同一个前馈 Actor 在相同实际状态、相同 observation/command、相同 normalizer 下，输出相同 mean action。两种 viewer 的行为区别主要来自 command 轨迹、后续状态与 reset/终止语义。**不能因为 Track 好，就认为独立恒定目标也已学会。**

checkpoint 的 `env_state` 只有 `loaded_4d_contract`，`Loaded4DTrackingManager.load_checkpoint_state()` 明确不恢复 mask、物理状态或 sampler buffer。保存的 viewer 配置中实际存在 ratio 1.0 与 0.0 的运行，未发现 Stage 1/2 checkpoint 把 Gen 覆盖回 Track 的证据。

## 4. 用户目标、覆盖范围与已知结果

这里的 `yaw` 指 `yaw_rate`，单位 rad/s，不是目标 yaw angle；height 是相对平面的 pelvis 高度，不是箱体高度。

| Viewer 目标 | Gen 训练覆盖 | 当前已知表现及结论 |
|---|---|---|
| Stand `[0,0,0,0.73]` | 明确的 standing 目标 | 用户报告其他已试 command 均失败；未保存该目标独立轨迹，不虚构站立稳定率 |
| Forward `[0.1,0,0,0.73]` | **范围内，必须优先处理** | 用户明确观察到单腿站立、没有实际前进；任务失败 |
| Forward `[0.4,0,0,0.73]` | Gen `vx` 范围外 | 用户报告已测试指令失败；不能以 Track 有近似速度为成功依据 |
| Forward `[0.8,0,0,0.73]` | Gen `vx` 范围外 | 同上；应在范围内控制通过后再验收 |
| Backward `[-0.1,0,0,0.73]` | 范围内 | 用户报告其他 motion cmd 失败；尚无单独速度曲线 |
| Left / Right `[0,±0.1,0,0.73]` | 训练边界 | 用户报告失败；需要分别测正负横向实际速度，不能用总体 MAE 替代 |
| Turn `[0,0,±0.2,0.73]` | 训练边界 | 用户报告失败；需要 heading 导数与目标比较，而非只看朝向发生变化 |
| Squat `[0,0,0,0.66]` | **训练边界，属于范围内** | 用户明确观察到单腿支撑、身体变化很小；未达到期望的持箱下蹲 |
| Rise / 连续切换 | 站立高度在范围内；连续过程另需验证 | 没有无 reset 的完整量化响应记录 |
| 多轴组合 | `axes=single` 未进行独立组合目标训练 | 过渡期短暂多轴非零不等于组合控制训练成功 |

用户描述中的“身体变化很小”还需区分两个量：pelvis 是否实际下降到目标，以及动作是否具备期望的双脚下蹲形态。**即使某个单腿姿态达到了 pelvis height，也不满足用户期望的下蹲动作。**反过来，仅观察身体摆幅不能推算具体 height MAE。

## 5. Command、观测与 viewer 通路审查

### 5.1 坐标系和时间索引

`managers/command/terms/loaded_4d.py:heading_state()` 统一用于 `reference_command()` 与 `measured_state()`：

- 从 pelvis quaternion 的前向轴求 heading；将 world linear velocity 转到 heading frame，得到 `vx/vy`。
- 令前向轴为 `f`，`df/dt = angular_velocity_world × f`，计算 `(f_x·df_y - f_y·df_x)/(f_x²+f_y²)`。这是 heading 的变化率，**不是直接取 world angular velocity 的 z 分量**。
- height 是 pelvis/root 的 z 减去 `scene.env_origins[:,2]`。原 tracking anchor 为 `torso_link`，没有被误用作 command 的 pelvis。

`begin_control_step()` 保存 Actor 所见的 `task_command` 为 `applied_command`。physics 后，Track reference 前进一次；Gen 发布下一步平滑后的 command。`command_tracking_reward()` 比较 post-physics measured 与刚刚执行的 **applied**，而非新的 target。当前源码未发现“指令使用箱体高度”“yaw 当 angle”“Gen reward 误用 reference measured state”或稳定的一帧 target/reward 错配。

### 5.2 Actor 的实际输入

ObservationManager 按 term name 排序。113D 的当前帧布局如下；切片为 Python 的左闭右开索引。

| Term | 切片 | 语义 |
|---|---|---|
| `actions` | 0:29 | 上一步 action |
| `base_ang_vel` | 29:32 | 实际 pelvis full body frame 角速度 |
| `base_lin_vel` | 32:35 | 实际 pelvis full body frame 线速度 |
| `dof_pos` / `dof_vel` | 35:64 / 64:93 | 实际关节状态 |
| `obj_lin_vel_b` | 93:96 | 实际箱体速度，相对实际 robot anchor 的姿态坐标系 |
| `obj_ori_b` / `obj_pos_b` | 96:102 / 102:105 | 实际箱体相对实际 robot anchor 的姿态/位置 |
| `pelvis_height` | 105:106 | 实际 pelvis 对地高度 |
| `projected_gravity` | 106:109 | 实际倾斜信息 |
| `task_command` | 109:113 | 当前平滑后的四维输入，scale 1、noise 0 |

`robot_ref_pos_w/robot_ref_quat_w` 的名字容易混淆：这里读的是 simulator 中 `torso_link` 的实际 rigid-body state。`obj_pos_b/obj_ori_b/obj_lin_vel_b` 并没有在 Gen 中使用冻结 reference 的箱体状态。Critic 的 reference privileged terms 与 Actor 的实际 object terms 是两类数据。

Actor 的实际 base velocity 是 full body frame，command 是 heading frame；网络同时有 projected gravity。当前没有显式 actual heading-yaw-rate scalar，但源码已提供恢复这些量所需的实际速度/姿态信息。可以研究增加派生量是否改善学习，**不能据此断言现有网络必然无法控制**；也不应先改 observation 维度使旧 checkpoint 失配。

### 5.3 Normalizer、探索与加载

Actor normalizer 的 command 均值为 `[0.05095,0.00393,-0.04409,0.69453]`，标准差为 `[0.15521,0.02611,0.38598,0.04754]`。`[0.1,0,0,0.73]` 标准化后约为 `[0.316,-0.151,0.114,0.746]`；`[0,0,0,0.66]` 约为 `[-0.328,-0.151,0.114,-0.726]`。这些范围内目标不是异常大的标准化输入。统计来自混合 Track/Gen 和训练历史，不能用于推断每种目标的响应增益。

checkpoint 包含两个网络、两个 optimizer 和两个 normalizer。`PPO.load()` 恢复它们；`get_inference_policy()` 使用恢复的 Actor normalizer，`update=False`，mean action 推理。最终 learned action std 的 mean/min/max 为 `0.08547 / 0.02037 / 0.16681`；首个 Stage 2 日志均值约 `0.03915`。没有探索方差归零的证据。训练 sampled action 与 viewer mean action 的差异仍需成对评估，不能仅凭 std 数值证明或排除闭环影响。

### 5.4 Viewer 与训练不完全相同的地方

`EvalViserCallback._setup_loaded_4d_controls()` 的预设与上表一致；选择预设只编辑字段，**点击 Apply 才提交**。回调通过普通数值队列交给评估线程，`on_pre_eval_env_step()` 调用已有 `set_command()`；该函数只写目标与 external mask，不写机器人/箱体物理状态。

| 差别 | 已确认影响 | 对本次失败的解释力 |
|---|---|---|
| 启动默认目标为 `[0,0,0,0.73]` | Gen reset 初始 command 来自 reference，随后转到默认 standing | 行走 frame 0 有初始运动速度；第一段站立过渡可能困难，但不能解释所有稳态目标失败 |
| 随机 vs external 过渡 | 随机前三维 0.8 s smoothstep；external 限速按外部范围宽度 / 0.8 s | 外部 `vx` 限速 2 m/s²；从 0 到 0.1 约 3 个 control step，即 0.06 s。过渡明显更快，应隔离检查；到达后长期不动不能只归因于这一短暂差别 |
| Height 过渡 | 两种方式均为 0.06 m/s | 0.73→0.66 的输入变化至少约 1.18 s；应在 Applied 到达后再算稳态误差 |
| horizon | 训练 15 s，交互 viewer 600 s | 长时间稳定性未训练；前 15 s 内的范围内失败仍是核心问题 |
| Reset | 训练随机初态；viewer 固定 motion/frame，并可能因失败自动 reset | 需要同时看 Reset 计数；反复初始化不能算连续控制 |
| 推理 | 训练 sampled Gaussian；viewer mean action、evaluation observation 设置 | 可能放大 train/eval 差异，目前没有成对 rollout 数据证实 |

UI 有 Target / Applied / Measured 读数，但它们没有持久保存。没有证据证明本次所有目标都停留在“只选预设未 Apply”；也不能在没有轨迹的情况下完全排除某一次操作未提交。正确做法是保留用户失败报告，并在后续量化时同时记录这三个量。

## 6. 训练日志：改善了哪些指标，没有证明哪些能力

读取 `RUN/events.out.tfevents.1795539364.han-Precision-3660.486900.0`：177217104 bytes，151 个 scalar tag，2265000 个 scalar sample，iteration 6000–20999。逐条读取并汇总整个文件，未发现非有限 scalar。下表是每个窗口 **1000 个 iteration 已记录标量的算术均值**，不是从原始物理轨迹重算的独立试验成功率。

### 6.1 四维平均误差

这些值来自 `Env/loaded_4d/{track,gen}/mae_*`。`Loaded4DTrackingManager._update_log_dict()` 在 reset 前，比较实际 measured 与本步 applied，按组平均；没有按 active axis、目标大小、motion 或过渡/稳态拆分。

| 组 / 指标 | 6000–6999 | 12000–12999 | 20000–20999 |
|---|---:|---:|---:|
| Gen `vx` MAE (m/s) | 0.05188 | 0.03666 | **0.03536** |
| Gen `vy` MAE (m/s) | 0.03500 | 0.02483 | **0.02528** |
| Gen yaw-rate MAE (rad/s) | 0.18713 | 0.15041 | **0.11986** |
| Gen height MAE (m) | 0.04132 | 0.01193 | **0.01378** |
| Track `vx` MAE (m/s) | 0.02139 | 0.02890 | 0.05140 |
| Track `vy` MAE (m/s) | 0.02612 | 0.02073 | 0.04020 |
| Track yaw-rate MAE (rad/s) | 0.18569 | 0.18698 | 0.28336 |
| Track height MAE (m) | 0.02841 | 0.02350 | 0.02500 |

Gen 整体 MAE 有改善，height 约从 4.13 cm 降到 1.38 cm。但这不能回答“`vx=0.1` 是否前进”或“`height=0.66` 是否双脚下蹲”。尤其 yaw-rate 末期总体 MAE 约 0.12 rad/s，而训练目标大多为零、非零目标幅度最大也只有 0.2 rad/s，这不构成良好的转向控制证据。

Track 末期部分 MAE 增大是真实统计现象，但同时 reference 采样分布改变，不能据此直接宣布逐条 reference 能力已遗忘。需固定 motion/start frame 才能与之前 checkpoint 公平比较。

### 6.2 存活、回报与 reference sampling

| 指标 | 6000–6999 | 12000–12999 | 20000–20999 |
|---|---:|---:|---:|
| Gen 平均 episode steps | 476.30 | 693.40 | 716.49 |
| Gen 平均 episode seconds (`steps×0.02`) | 9.526 | 13.868 | **14.330** |
| Gen 平均 episode return | 54.174 | 83.447 | **86.146** |
| Track 平均 episode seconds | 3.866 | 3.535 | 1.714 |
| Track 平均 episode return | 23.904 | 21.995 | 10.508 |
| adaptive sampler 最大 motion 概率 | 0.3335 | 0.4098 | 0.8460 |
| adaptive sampler 有效 bins | 17.005 | 14.702 | 2.737 |

`sampleweight.txt` 的最后记录 step 20900 直接显示：Track sampling motion 1 概率 `0.83561456`，其中后半 bin（frame 50–109）的条件概率 `0.93110806`。行走只有 2.18 s，后半段起点更接近 clip end。**Track 回报与 episode 长度降低有明确的采样/片段长度混杂因素。**这些是 sampler 概率，不是三条 motion 的成功率，也不是其训练时间占比。

Gen 存活接近 15 s 上限，与用户看到的单腿站住可以同时成立：物理存活不等于指令成功。这里不把 14.33/15 换算成成功率。

### 6.3 现有日志容易造成的误读

- `Env/loaded_4d/gen_tracking_baseline_per_second` 恒为 **1.75**、`...per_step` 恒为 **0.035**，是 50% Gen 占比后的全环境平均。单个 Gen 的真实 tracking 常数是 **3.5/s、0.07/step**。
- `HybridBadTracking` 的旧 Gen low-root/ref-z/object-z/orientation 检查在 Loaded-4D preset 中被关闭。因此相关旧 termination tag 为 0，**不等于没有物理失败，更不等于 motion cmd 成功**。
- 末期 `Termination/Env/physical_failure_frac≈0.000045` 是每个 control step 中所有环境的失败比例，既不是 Gen 独立概率，也不是 episode 失败率。clip-finished fraction≈0.005882、timeout fraction≈0.000653 也须按此口径理解。
- `Episode/rew_command_*` 是 reset episode reward 按固定 max episode seconds 归一化后的混合统计；Track 的这三项为 0，且 Track reset 更频繁。不能把该曲线下降直接解释为 Gen reward 下降。
- `wbt_manager.py:HYBRID_GEN_REWARD_TERMS` 仍列旧 endpoint reward，没有登记三个新 `command_*`。所以这里没有完整的 Gen-only 各项 command reward episode 拆分；已有 Gen 总回报仍然可用。
- `evaluate_loaded_4d.py` 的 `completed_trials` 对 Gen 表示达到 horizon 而没有提前终止；源码没有 command error 成功阈值。**即使未来输出 completed，也仍需检查误差、响应与姿态。**

## 7. 为什么“站住不动”和“单腿下蹲”可能得到高分

### 7.1 当前 Gen 的优化目标

由保存的 reward 配置、`reward/terms/wbt.py:_mask_track_envs()` 和 `envs/wbt/loaded_4d.py:command_tracking_reward()` 可得，在未触发物理失败时：

```text
r_velocity = exp(-((vx_actual-vx_cmd)^2 + (vy_actual-vy_cmd)^2) / 0.20^2)
r_yaw      = exp(-(yaw_rate_actual-yaw_rate_cmd)^2 / 0.30^2)
r_height   = exp(-(height_actual-height_cmd)^2 / 0.05^2)

Gen positive reward rate = 3.5 + r_velocity + r_yaw + r_height
Gen total step reward = 0.02 × (above + regularization)
```

8 项原 tracking reward 的 weight 合计为 7，Gen 每项 raw 常数 0.5，因此得到 3.5/s。三个新 reward 的 weight 都是 1，正向上限为 6.5/s。regularization 为 action-rate weight −0.1、joint-limit −10、undesired-contact −0.1。

以下反例假设箱体/姿态通过物理 gate，忽略 regularization；除列出的误差外，其余目标均满足。百分比是**正向奖励上限之比，不是任务成功率或实际测得回报**。

| 未完成的目标 | 假设 actual | 该目标 reward | Gen 总正向 rate | 上限占比 |
|---|---|---:|---:|---:|
| `vx=0.1` | `vx=0` | 0.7788 | 6.2788 | **96.60%** |
| `vy=0.1` | `vy=0` | 0.7788 | 6.2788 | 96.60% |
| `yaw_rate=0.2` | `yaw_rate=0` | 0.6412 | 6.1412 | 94.48% |
| `height=0.66` | 一直保持 `height=0.73` | 0.1409 | 5.6409 | 86.78% |

前进 0.1 m/s 时，完全消除速度误差只增加约 **0.2212/s，即 0.004424/control step**。移动同时可能增加 action-rate/contact cost，形成保持姿态的局部最优是合理假设；但没有分目标 reward breakdown，尚不能证明实际策略每一步都因为这项 cost 放弃移动。

不能把问题简化为“只要删掉常数就好”：在固定生存时长和相同终止的比较里，加同一个常数不改变两种动作的即时 reward 差，理想 value baseline 也会抵消与动作无关的部分。正的常数确实抬高总回报，在不同生存时长下提供存活收益，并使看总分的验收很容易误判。**sigma/weight 与任务尺度、目标分布、动作约束和实际闭环响应需要一起分析，不能仅凭 96.6% 决定删除常数。**

### 7.2 Zero-heavy 目标让总体 MAE 难以验收控制

`_sample_gen_targets()` 先均匀选一个 active axis，再以 20% 概率覆盖为 stand。因此在“已到达采样目标”的理想分布中：

- 每一个速度/转速轴约 80% 的目标为 0；每个轴仅约 20% 的目标来自该轴的非零连续采样。
- height 约 80% 为 standing 0.73；约 20% 来自 0.66–0.73。

假设策略永远保持实际 `[0,0,0,0.73]`，则在这个**理论稳态目标分布**下，总体 MAE 已经只有：

| 维度 | 静止基线 MAE | 计算 |
|---|---:|---|
| `vx` | 0.015 m/s | `0.2 × 0.15/2` |
| `vy` | 0.010 m/s | `0.2 × 0.10/2` |
| yaw-rate | 0.020 rad/s | `0.2 × 0.20/2` |
| height | 0.007 m | `0.2 × 0.07/2` |

同一假设下，三个 reward 的期望约为 `0.95258 / 0.97394 / 0.92056`，加常数得到 `6.34708/s`，约为正向上限的 **97.65%**。计算使用均匀分布上 `exp(-(error/sigma)^2)` 的积分。

这是解析静止基线，不是现有模型实验结果；reset 时从 reference command 开始、过渡插值、实际姿态扰动和失败会改变真实分布。它说明当前总体误差和总回报并不能区分“学会每个目标”和“大多数时间保持稳定”。末期日志 MAE 也不能与这个解析基线直接组成严格 benchmark 排名。

### 7.3 单腿姿态为什么没有被排除

`physical_failure()` 检查：数值/四元数有效、关节硬限位外 0.03 rad、pelvis height <0.40、up.z <0.20、heading 投影奇异、箱体最低 collision corner 离地 <=0.02、箱体与 pelvis 距离 >1.0 m。

这些检查没有要求站立/下蹲双脚接触，没有检查两手握持或箱体相对双手几何，没有限制躯干/箱体以何种方式保持稳定。`BodyGroupProximity` 只在两脚位置距离 <0.05 m 时终止，**不是双脚必须着地**。`UndesiredContacts` 处罚不期望的接触，却不奖励所期望的双脚支撑。

因此在 Gen 中，只要 actual height、速度等接近目标且保持箱体离地，一个单腿姿态可以获得与双脚下蹲相近甚至相同的任务奖励。Track 里的关节/身体姿态与箱体 tracking 在 Gen 中已成为常数，shared weights 本身不是动作风格保持约束。

也不能给所有 Gen command 强制双脚永久接触：正常行走需要单脚支撑。若后续增加约束，应局限于近零平移/转速的 standing/squat 工况，并防止脚尖轻触或滑脚规避；不应引入完整搬运 phase state machine。

## 8. 原因分级与尚不能下的结论

| 判断 | 证据等级 | 对本次问题的含义 |
|---|---|---|
| 范围内 `vx=0.1`、下蹲失败 | **用户实际观察** | Gen 当前验收失败，必须优先处理；不是单纯范围外问题 |
| Gen 缺少 standing/squat 支撑方式和动作形态约束 | **源码确认** | 单腿姿态没有被任务定义排除；与用户现象直接一致 |
| 宽容 reward + zero-heavy targets 容许静止高分 | **源码与解析计算确认** | 高回报/低平均误差不足以证明控制；静止吸引态是优先假设 |
| 模型实际因上述原因形成单腿局部最优 | **合理推断** | 仍需 per-command 轨迹和一次受控修正验证，不能写成唯一已确认根因 |
| Track 到低速纯前进的泛化缺口 | **数据事实 + 学习假设** | Track 行走约 0.293–0.787 m/s，不能自动提供 0.1 的独立慢走；但 0.1 已在 Gen 训练，应由 Gen 学会 |
| 训练/手动过渡、RSI/horizon、sampled/mean 差异 | **实现差异已确认，影响待测** | 可能造成部分分布偏移；不能用来抹去范围内失败 |
| normalizer 没加载、command 没进 Actor、Gen 被恢复成 Track、yaw 用错定义 | **本次未发现支持证据** | 当前静态通路和 checkpoint 不支持这些归因，优先级低于任务奖励缺口 |
| PPO/学习率/physics 是唯一根因 | **未证实** | 有效训练发生、标量有限、Track 能力保留；不应首先重写 PPO、action、PD 或同时改物理频率 |

末期 adaptive learning rate 约 1e−5，KL 窗口均值约 0.0114；这证明实际 optimizer 状态与配置初值不同，但不能从这些数字单独诊断 PPO 失败。损失下降同样不代表四维控制目标实现。

## 9. 下一步最小诊断：先把 failure 测准

以下是**待执行的诊断建议**，本次没有运行。保持当前 checkpoint、训练文件和原始日志不变，结果写入新的审查目录。

### 9.1 范围内目标优先，逐条评估

先对 `stand`、`vx=0.1`、`vx=-0.1`、`vy=±0.1`、`yaw_rate=±0.2`、`height=0.66` 做单目标、15 s horizon、相同 start frame 的确定性评估。优先比较 motion 0 与 motion 1 的 frame 0，以区分初态影响；随后必要时补 motion 2。不要一开始把所有目标连成大序列，因为前段失败会导致后段根本没有执行。

已有 `scripts_run/evaluate_loaded_4d.py --sequence` 可接受只有一行的 JSON，并记录 10 Hz 的 target/applied/measured、终止原因、reference frame、箱体 clearance。它已经能判断是否“没有前进”；无需先改 PPO 或环境框架。后续整理时应按目标分别汇总，而不是只看 `completed_trials`。

必须记录或计算：

1. Applied 是否到达 Target；排除输入 ramp 后，实际均值、MAE/P95、正负方向、净位移/heading 变化。`vx=0.1` 的判据必须包含实际有正向运动；近零响应不接受。
2. Height 的实际起止值和变化量；0.73→0.66 的目标差为 7 cm。区分达到高度但姿态不对，与根本没降低 pelvis。
3. 每次真实 termination/reset、无 reset 的持续时长；不能把多段重置后存活拼成成功。
4. 站立/下蹲左右脚接触、脚高度/滑动、躯干姿态，以及双手—箱体相对几何。现有 evaluator 没有完整保存这些字段，量化单腿和 grasp 需要小范围补充诊断读数。

验收时先给出目标条件化响应，再制定误差门槛。若采用候选门槛，例如 `vx=0.1` 稳态 MAE≤0.03 m/s、height 稳态 MAE≤0.02 m，应标为新的审查标准，并同时满足姿态和无重置条件；不能由训练总体 MAE 反推“已经通过”。

### 9.2 两个最有区分力的对照

- **相同 reference command 序列的 Track/Gen 对照**：相同物理初态、冻结 normalizer、同样 mean action，给 Gen 逐帧回放同一 reference-derived command。必须让 Actor 实际看到的 command 完全一致；简单把序列送进现有 external slew limiter 不构成严格等价。若一致输入下轨迹仍分叉，再追查 mode 对 physics/observation/reset 的隐藏依赖。这只需评估层的短诊断，不代表给 Gen 恢复逐帧 tracking 奖励。
- **范围内固定目标的 deterministic/stochastic 对照**：复用 evaluator 的 `--stochastic`，比较同一初态和目标。如果 sampled action 明显更好，才进一步研究 train/eval 问题；不先增加噪声掩盖 deterministic 失败。

必要时再用同样评估比较现成 `model_09000/12000/15000/18000.pt`，定位是否曾有 Gen 响应后退化。没有这些结果之前，不声称 20999 比中间 checkpoint 更好或更差。

## 10. 后续修改的最小范围（本次未实施）

| 实际文件 | 相关函数/类 | 建议及依赖 | 验证方式 |
|---|---|---|---|
| `REPO/scripts_run/evaluate_loaded_4d.py` | `run_trial`、`summarize_samples`、`evaluate` | 先直接复用固定单目标与 trace；若需量化支撑/抓握，仅增加足底/箱体状态记录及按目标的摘要。不要把 horizon_complete 当 task_success | 同一模型、初态和目标，验证 Target/Applied/Measured 对齐；首次失败停止；无隐藏 reset |
| `SRC/envs/wbt/loaded_4d.py` | `_update_log_dict`、`command_tracking_reward`、`physical_metrics` | 优先补 active-axis/稳态诊断；确认问题后再选一个局部 reward/约束改动。新 carry/stance 约束须依赖 actual state，不引用冻结 reference | 静止与正确运动的 reward 对比；单腿与双脚 standing/squat；正常单脚支撑行走不被误罚 |
| `SRC/config_values/wbt/g1/loaded_4d.py` | `loaded_4d_preset` | 若诊断支持，单独调整 reward sigma/weight 或 command sampler 分布，保留本次 baseline；先修范围内响应，再扩大到 pair/combined | 不同时改物理/网络/action；原 Stage 1/Track 回归与固定范围内目标复测 |
| `SRC/managers/command/terms/loaded_4d.py` | `_sample_gen_targets`、`_advance_gen_commands` | 仅当需要时统一 train/manual 过渡或改善目标覆盖；`axes`/范围已可配置的部分不用增加新系统 | 更新 command 不写物理状态；既有时序和接口不变；相同目标记录实际过渡 |
| `SRC/envs/wbt/wbt_manager.py` | `HYBRID_GEN_REWARD_TERMS`、hybrid metric helpers | 可选：补齐新 reward 的 Gen-only episode 统计；也可在 Loaded-4D 内局部记录以保持共享文件不变 | 确认 denominator、单位、mask；原 endpoint 统计保持正确 |

不建议作为第一步修改：PPO、Actor/Critic MLP、action parameterization、PD gains、physics dt、motion loader、reference NPZ、上游 presets。也不建议为了复制 reference 动作给 Gen 加完整 phase/pose 输入、AMP、motion generator 或 pickup/putdown state machine。

推荐顺序：**固定目标量化 → 检查奖励反例与支撑缺口 → 一项局部修正 → 短训练和同条件评估 → 范围内通过后再做组合/长时间切换**。当前没有证据支持直接延长原配置训练或先尝试更高速度就会解决问题。

## 11. 证据索引与离线复核

除第 2 节指定的 run/config/checkpoint 与 `sampleweight.txt` 外，关键结论对应以下现有源码：

| 相对 `SRC` 的文件（另注明 `REPO` 时除外） | 核查项 |
|---|---|
| `config_values/wbt/g1/loaded_4d.py:loaded_4d_preset` | command ranges、三项 reward、termination、randomization、Stage 1/2 接口 |
| `managers/command/terms/loaded_4d.py:Loaded4DCommand` | reference/actual command、reset、target sampling、external interface、时序、Gen clip 不结束 |
| `managers/command/terms/loaded_4d.py:heading_state` | heading velocity、严格 yaw-rate、pelvis height |
| `envs/wbt/loaded_4d.py:Loaded4DTrackingManager` | command reward、物理 gate、reset 前 snapshot、分组 MAE、checkpoint contract |
| `managers/command/terms/wbt.py:MotionCommand` | 实际 `robot_ref_*`、reference loader/reset、adaptive sampling |
| `managers/reward/terms/wbt.py:_mask_track_envs` | Gen tracking 常数 0.5 |
| `managers/reward/manager.py:RewardManager.compute/reset` | weight×dt、episode reward 归一化 |
| `managers/observation/terms/wbt.py` | 实际 robot/object observation 与 hybrid reference mask |
| `managers/termination/terms/wbt.py:HybridBadTracking/BodyGroupProximity` | reference termination 与脚距离检查 |
| `envs/wbt/wbt_manager.py`、`envs/base_task/base_task.py` | reward/termination 日志口径和 physics/reward/reset 顺序 |
| `agents/ppo/ppo.py:PPO.load/get_inference_policy` | 网络/normalizer/optimizer/std 恢复、deterministic 推理 |
| `agents/callbacks/viser_callback.py:EvalViserCallback` | 预设、Apply、外部目标、显示和 reset 读数 |
| `REPO/scripts_run/evaluate_loaded_4d.py` | CPU 数据校验、连续 trial、trace、任务成功判据的缺口 |

下面的离线命令可复核第 6 节的核心窗口均值，只读 TensorBoard，不启动仿真、不修改 run。使用已有 runtime，无需安装依赖：

```bash
cd /home/han/pyx/HMI_clean
source /home/han/pyx/HMI_clean_runtime/activate.sh

PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from collections import defaultdict
from pathlib import Path
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

run = Path('/home/han/pyx/HMI_clean_runs/loaded_4d/stage2/WholeBodyTracking/20261124_165511-g1_loaded_4d_stage2_joint-locomotion')
windows = [(6000, 6999), (12000, 12999), (20000, 20999)]
prefixes = ('Env/loaded_4d/', 'Hybrid/Episode/episode_length/',
            'Hybrid/Episode/mean_reward/')
sums, counts = defaultdict(float), defaultdict(int)
for event in LegacyEventFileLoader(str(next(run.glob('events.out.tfevents.*')))).Load():
    for value in event.summary.value:
        if not value.HasField('simple_value') or not value.tag.startswith(prefixes):
            continue
        for lo, hi in windows:
            if lo <= event.step <= hi:
                key = (value.tag, lo, hi)
                sums[key] += value.simple_value
                counts[key] += 1
for key in sorted(sums):
    print(key, 'count=', counts[key], 'mean=', sums[key] / counts[key])
PY
```

本次文档交付只记录已完成的审查和后续建议。**范围内 Gen 控制失败已登记；精确逐指令响应和根因干预仍待独立评估。**
