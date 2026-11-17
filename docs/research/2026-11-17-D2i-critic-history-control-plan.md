# DLS D2i — Critic 观测历史是否有价值（预注册，尚未执行）

**日期**：2026-11-17。承接 D2h 的独立验证结果，不涉及 A/B/C Actor 训练，也不将其当作通信机制证据。

## 研究问题与为什么值得做

原 A/B/C Actor 的控制动作依赖循环状态 `h_t`，但 D2b–D2h 的 `IdenticalCritic` 只接收单步 259 维观测 `o_t`。低解释方差可能是欠训练，也可能是未观察到足够的历史。成熟 SB3-Contrib RecurrentPPO 支持 LSTM 价值头，但不可直接替代我们的自定义 Actor（参见官方框架调研）。

固定已经保存的 D2h train/validation 数据，只检验：

**用完全相同的 Critic 参数和监督数据，让它保留最近 64 步的历史，相比每一步清除隐藏状态，是否能够更准确预测实际后续折扣回报？**

## 设计（只动一个点）

- 采用相同 `GRUCell(259,24) + Linear(24,1)` Critic。训练参数共 `3×24×259+3×24×24+6×24+25=20,545`，两组严格一致；相比旧 20,865 参数 MLP 差异约 -1.53%。这不是重新设计 A/B/C。
- **History-ON**：每个长度 64 的窗口内 `h_t=GRUCell(o_t,h_{t-1})`。
- **History-OFF**：相同网络和参数量，但每一步都令 `h_{t-1}=0`，即没有时间历史。
- 两组使用完全同一初始权重、相同优化器设置/步数、相同 minibatch 序列和 mask；仅 ON/OFF 的状态传递不同。
- 直接复用 `/mnt/datasets/lxb/dls/rollouts/d2h-v1/` 数据；每个 A/B/C actor 单独使用 train 4600–4623、val 4700–4715 的成对轨迹。**不重新运行 MetaDrive，不查询专家，不改变 Actor 或环境。**
- 严格按 `episode_offsets` 截取序列，不跨 episode、不跨 train/val 拼接；64 步窗口、窗口开头隐藏状态归零；小于 64 的尾部 padding 不计损失。原 D2h 标记为 `censored_time_limit` 的整条 episode 不进入 supervised loss。
- 同样的 raw discount-return MSE，AdamW、固定 epoch 和相同 optimizer update 预算；不根据验证表现选择 epoch 或系数。用固定 seed 确保两组的 minibatch 采样序列一致。
- 对每个 A/B/C：报告 validation 的 EV、RMSE、MAE、Pearson，以及按 episode 汇总的 MAE；与常数 train-mean baseline 对照。通过 `episode` 而非 `step` 做 paired 不确定性统计，避免时间相关性制造虚假精度。

## 事前判断标准

- 如果 History-ON 在至少 2 个 A/B/C 占据分布上实现 **≥5% 的验证 RMSE 相对下降**，并且 EV 为正，认为近期历史对价值学习有可用信号。下一步在成熟 PPO 管线里加入匹配的 recurrent critic，并重新做同等采样与计算预算控制。
- 如果 History-ON 没有清晰增益，不说明长期历史没有用（仅看了 64 步、24/16 小量场景）；优先考虑目标回报尺度、时间限制处理、样本量与 reward 噪声，同时核查官方 PPO 基线。
- 不允许只报告 A/B 或在看到结果后改变序列长度与成功阈值；C 的失败同样保留。
- **即使 ON 显著改善 Critic，也不证明 PPO 性能会提升**。必须在官方 PPO 兼容性通过后进行同场景闭环训练验证。

## 资源与下一步

- 没有额外环境数据与安装开销；使用已完成 D2h 的数据盘轨迹，现有 PyTorch 2.5.1 与 4090。
- 与此独立的工程任务：在隔离环境中先做 SB3-Contrib / TorchRL 兼容性 dry run 和 MetaDrive 0.4.3 环境 API smoke；不得直接对 `dls-p0` 运行未固定版本的 `pip install`。
- **准备阶段终点**：参考 PPO 在当前连续动作交通场景能够稳定优化；A/B/C 都通过循环隐状态、价值目标和公平预算审计；才能进入通信干预的主科学实验。此计划只是候选诊断，尚无 D2i 实测结果。

相关记录：[D2h 与官方框架精修](2026-11-17-official-RL-stack-audit.md)。
