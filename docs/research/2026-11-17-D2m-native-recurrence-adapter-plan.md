# D2m — A/B/C 原生循环控制器的异步并行状态兼容性（接口准则）

日期：2026-11-17。D2l 正式 PPO 吞吐测试成功；D2m 从 D1 原始冻结权重开始。

## 与真正科学问题的关系

DLS P0 主问题不是「哪个 PPO 框架最好」，而是 B（仅物理反馈）与 C（直接可学习模块通信）是否学出可复用的分工和因果补偿。D1 的 A/B/C 已有基本 BC 驾驶能力。为把经验证的官方 SB3-Contrib 采样和优化系统应用于这些**已有的模型**，必须保持相同网络和状态递推，否则 D2n 的对照就失去科学意义。

## 为什么标准 SB3 的 LSTM 不能直接替换

[SB3-Contrib RecurrentPPO 2.6.0 源码](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.6.0/sb3_contrib/ppo_recurrent/ppo_recurrent.py) 的 `_setup_model()` 由 `policy.lstm_actor` 创建对应尺寸的 `RecurrentRolloutBuffer`，在 rollout 时调用 `policy.forward()`，更新时调用 `policy.evaluate_actions()`。默认 [RecurrentActorCriticPolicy](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.6.0/sb3_contrib/common/recurrent/policies.py) 假定 `(h,c)` LSTM 记忆形状；我们的原生控制器是 A 的单 GRUCell 或 B/C 的4个GRUCell，还额外有16维慢环状态 `z` 和每5步更新一次的**独立时钟**。

在64个 worker 异步重置的情况下，使用一个共享的整数时间 `t` 会错误更新一部分环境的慢环（尤其新 episode 只重置一部分环境时），出现误差也无法靠 PPO 的官方稳定性来掩盖。

## D2m 一项最小可证伪测试

不修改 `src/dls_p0.py` 公开接口，增加 `src/dls_vector_state_adapter.py`：

- 每 worker 状态 `(h,z,phase)`，其中 `phase∈{0,1,2,3,4}`；每次根据 episode_starts 只重置该 worker 的状态和时钟；
- 每一步根据 phase 将同相位环境打成最多5批，逐批调用**原版** `actor.step(obs,(h,z),t=phase)`，再按输入环境索引恢复动作与新状态。模型/参数不变，包含相同模块通信计算；
- 提供无损 `flatten/unflatten` 状态编码，供后续标准 recurrent rollout buffer 适配；
- **正确性**：对 A/B/C 原 D1 seed42 权重，以 48 个异步环境、37 步随机 observation 与可复现异步 reset，与每环境独立的单体循环执行比较，动作、GRU 隐状态、慢环状态误差均低于2e-5；
- **概率重放**：同一缓存初始状态、输入、潜在动作，重复前向计算所得对数概率 ratio 应为1（误差小于3e-5）；
- **可微性**：经分相位 index_copy 后，PPO 时间递推依旧能使 A/B/C 各模型梯度有限且非零；
- **资源**：只读原 D1 BC checkpoint，CPU 脚本，无 MetaDrive 仿真、无新数据与 PPO optimizer 更新。验证后校验 checkpoint SHA-256 未变化。

## 成功的含义与下一步

D2m 只证明**状态接口的数学等价与梯度保持**，**不能冒充已完成 SB3 自定义策略接入**。下一阶段必须实现符合 SB3-Contrib `forward/evaluate_actions/get_distribution/predict_values` 与 buffer 时序的兼容策略（或经过核验改用 TorchRL 的 PPO组件），验证真实 rollout 的 PPO probability ratio=1，再投入正式 A/B/C 训练。

当完成上述最小接入 smoke，即结束扩展性准备工作；不再为了完善通用 PPO 额外开系列实验。科学阶段首先验证已有 BC 功能保留和三组同样环境预算下的闭环学习，再做 D3 frozen-weight 模块干预。

GitHub workflow: `.github/workflows/d2m-vector-state-smoke.yml`。
