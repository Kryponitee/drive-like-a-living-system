# DLS D2g — 冻结策略的 Critic Value Calibration 诊断（预注册）

日期：2026-11-17。D2f 结果已获得，本文件在 D2g 运行前创建。

## 关键问题

D2f 在 D1 9822 步验证集上的动作保真显著改善，但在全新场景上的闭环收益只有 B 显著表现为正向，A/C 没有跨架构一致收益。PPO 最初使用**随机初始化 Critic**，总预算仅 4096 个环境步，需检查其估值是否可用于有效的优势估计。

**假设：D2d / D2f PPO 的 Critic 估值与实际后续奖励缺乏对齐，会给 actor 提供高噪声或误导性的训练方向。**

本项只是测量预测与真实回报的关系，不训练，不修改模型，不作为因果定论。

## 一个点与测量

- 冻结 D2d、D2f 各 A/B/C 的 **6 对 actor+critic 权重**，采用 PPO 训练结束时保存的 `log_std`，动作在 tanh 前分布采样，状态维度 259 与归一化方式不变。
- 全新 MetaDrive seeds **4500–4523**，固定 `map=3`、`traffic_density=0.10`、`horizon=800`。6 个独立 CPU 仿真进程，保留每场景随机数流使不同 actor/phase 使用相同基础噪声序列。
- 每时刻记录 `V_\phi(s_t)`、即时奖励 `r_t`、是否真实终止/截断、终止时成功/碰撞/越界，以及下个状态。
- 以 PPO 的 `γ=0.99` 计算后续折扣回报 `G_t = r_t + γ G_{t+1}`；若仅时间上限截断，按 `V(s_T)` bootstrap，并单独统计这类比例。
- 对每组报告 **value vs realized return** 的 Pearson、解释方差 `1-Var(G-V)/Var(G)`、value MAE、平均一步 TD 绝对残差与开始状态 Value/Return 的 across-episode Pearson。它们是 critic 校准/噪声的诊断代理。
- **判据：** 如果六组的 explained variance 接近或小于零、value MAE 显著大于奖励变化尺度且初始 V/回报关联弱，则优先研究 Critic 预训练或 rollout 价值回归；如果 Critic 已能追踪回报，下一步不再把短预算 PPO 退化归咎于 Critic，考虑优化步长或在当前 BC 基础上提早开展 D3 模块干预。
- **限制：** 单次随机 rollout 的 realized return 具有高方差；同一 episode 的时间步非独立，critic 回归并非 causal gradient correctness 检查；D2d/D2f actor 不同，跨阶段估值变化同时受占据分布影响。不能从负 explained variance 直接证明“价值网络导致了 PPO 失败”。

运行只读：复用 D2d、D2f checkpoint 和 MetaDrive 环境，无新 GPU 训练、无 D1 数据重采集，所有诊断在数据盘 `/mnt/datasets/lxb/dls/reports/d2g/`。
