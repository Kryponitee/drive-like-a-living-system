# D2m — A/B/C 异步状态与官方 RecurrentPPO 接入结果

日期：2026-11-17。两阶段均已完成：

- [D2m 48 个异步状态时钟等价性 #44030866838](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44030866838)：**SUCCESS**。
- [D2m+ 官方 RecurrentPPO 自定义策略与 MetaDrive 训练 #44031524331](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44031524331)：**SUCCESS**。

## 官方框架与已有科学模型的连接

SB3-Contrib RecurrentPPO 2.6.0 的 `RecurrentRolloutBuffer` 固定接受类似 LSTM 的 `(h,c)` 状态，`_setup_model()` 从 `policy.lstm_actor.num_layers,hidden_size` 计算缓冲区尺寸。原 DLS A 是单 GRUCell +16维慢状态，B/C 为 4 个 GRUCell +16维慢状态，并使用 **每5步更新慢状态** 的独立时钟。异步 vectorized MetaDrive worker 不能共用同一个 `t`。

我们保留原 `src/dls_p0.py` 不变：

- `src/dls_vector_state_adapter.py`：将每 worker 的 `(h,z,phase)` 无损打包，并在5个时钟相位组内执行原 `actor.step`；每个 worker 只在自己的 episode start 时重置。这不增加网络模型参数。
- `src/dls_sb3_recurrent_policy.py`：使用 SB3-Contrib 官方的 `RecurrentActorCriticPolicy` 接口，采用**无参数的 LSTM 状态形状适配器**以复用其 `RecurrentRolloutBuffer` 和 PPO/GAE 更新，实际 actor 仍为原有 A/B/C。官方策略接口 `forward/get_distribution/predict_values/evaluate_actions` 均使用真实的模块控制器。Critic 使用统一的已有 `IdenticalCritic`。
- 采用官方 `SquashedDiagGaussianDistribution`，初始 tanh 前标准差 `σ=0.05`。输入仍由冻结 D1 训练统计独立归一化，不能在 VecNormalize 再次规范化观察，否则旧 BC 能力会被破坏；训练中可以复用官方 reward normalization。

## 确认事实

**D2m 分相位与递推等价性**，48 异步环境×37步，原 BC seed42：

| 模型 | action 最大误差 | hidden/slow 最大误差 | PPO 相同动作 logprob ratio误差 |
|---|---:|---:|---:|
| A | 2.68e-7 | 3.87e-7 | 0 |
| B | 3.58e-7 | 6.56e-7 | 0 |
| C | 2.38e-7 | 4.77e-7 | 0 |

梯度非零且有限，快环/慢环状态打包与解包完全一致，B/C 原训练参数量保持严格相等。原 checkpoint SHA-256 保持不变。

**D2m+ 官方真实训练 smoke**：先分别实例化 A/B/C 并确认 deterministic action 精确匹配原 BC，随后用 B 模型在**4 个真实 MetaDrive 子进程中做 512 环境步、4 个 rollouts 的官方 RecurrentPPO 训练**：

- 四个 rollout 在任何 optimizer 更新前，以官方 recurrent buffer 的时间序列/episode mask 重新计算旧策略 log probability，**最大 ratio 偏差2.62e-5**；
- SB3-Contrib 官方 PPO 进行了实际 actor 和 critic 更新，参数 L1 变化分别为 **27.693** 和 **23.156**，均不是零；
- 没有重写 SB3 的收集器/GAE/PPO 更新逻辑，没有重新采集专家示范，没有修改原始 D1/D2 权重，整个 DLS 运行环境不变。

## 研究结论及严格限制

**工程准备可以收口**：已验证64 CPU workers 完整 PPO 的加速（见 [D2l](2026-11-17-D2l-full-PPO-worker-results.md)），已验证 A/B/C 原生异步控制时钟，在官方 RecurrentPPO 中实际采样和更新，并保持 B/C 容量对照接口。

但短 smoke **并不能证明**：完整 64-worker A/B/C 训练已经具有良好控制能力、反馈协同模块已经出现，或通信边有因果效果。

据此已直接进入 **D2n-A（正式 BC 初始化的 A/B/C 对照训练）**，预注册在 [D2n 协议](2026-11-17-D2n-A-official-BC-PPO-protocol.md)。今后不因发现新的 PPO 实现选择而推迟 D3 研究；若 D2n 失败，仅排除确切的正确性/能力问题，保留 D1 BC 冻结策略作为独立 D3 机制诊断入口。
