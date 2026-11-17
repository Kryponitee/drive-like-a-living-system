# DLS D2i-H — 等参数 Critic 历史 ON/OFF 验证结果

日期：2026-11-17。测试已完成，GitHub Actions [#44024937004](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44024937004) **SUCCESS**；预注册方案：[D2i Critic History Plan](2026-11-17-D2i-critic-history-control-plan.md)。

## 目的和控制

D2g 的 Critic 价值解释方差接近零，D2h 监督学习使 A/B 的预测误差下降但 C 泛化差。本项仅问一个问题：

**在相同网络参数、训练预算和 D2h 状态回报对照数据下，携带最近 64 步的历史是否有独立预测价值？**

- 完全复用 D2h 已存的完整回报监督（训练 seeds4600–4623 / 验证 seeds4700–4715），无需再运行 MetaDrive；
- 三个 BC Actor 全程冻结；对各个 A/B/C 的独立访问状态分布分别做训练与验证；
- ON：GRU 64 步窗口内携带 `h_{t-1}`；OFF：每一步输入相同 GRU 参数但 `h_{t-1}=0`；
- 每个 GRU Critic `259→GRU(24)→Linear(1)` 共 **20,545 参数**。具体使用 `nn.GRU` 执行数学等价的 GRUCell 递推，OFF 将每个输入变为长度 1 的序列，避免大量 Python 级 GRUCell 循环；
- ON/OFF 初始权重逐 tensor 完全一致，同样训练 minibatch 顺序、AdamW 和固定 **50 epochs**；
- 预测仍在原始回报尺度计算 MSE/RMSE/EV。两组共享训练集回报均值/标准差作为最后一层的固定仿射标定（不引入可训练参数）；不根据验证集挑 epoch。
- 每个 episode 独立切 64 步窗口，窗口开始隐藏状态为 0，不跨 episode，短尾只计有效步；删失 episode 不用于监督。
- 不同 Actor 的状态占据分布不同，因此仅解释 **每个分布内部 ON 与 OFF 的配对差异**，不能跨 Actor 的 RMSE 排名。

## 固定 50 epoch 的独立验证结果

| Actor 状态分布 | History ON RMSE | History OFF RMSE | ON 相对降低 | ON EV | OFF EV |
|---|---:|---:|---:|---:|---:|
| A | 17.125 | 17.184 | **0.35%** | 0.481 | 0.483 |
| B | 21.236 | **19.598** | **−8.36%** | 0.294 | 0.377 |
| C | 22.324 | **22.252** | **−0.32%** | 0.245 | 0.234 |

预注册的正向证据判据：至少 2/3 分布上，ON 相对 OFF 的验证 RMSE 降低 >=5%，且 ON EV 为正。**最终 0/3 符合，未通过。**

训练中期出现过暂时性 ON 优势（第10 epoch：A 18.356 vs OFF19.728；B 20.003 vs21.176；C 21.441 vs22.397），但到固定训练终点并不保持。不能事后从中选择最好的检查点改称证据。

## 有意义的结论（含限制）

- **拒绝当前具体假说**：在短 64 步历史、当前 D2h 数据与 50 epoch 训练下，没有跨 A/B/C 稳定收益，不能将近零 EV 主要归咎于「Critic 缺少循环记忆」。
- **不否定一般时序信息价值**：长于64步的记忆、不同损失、训练预算和真实 on-policy PPO 中的 value regression 可能不同；上述未检验。
- A/B/C 的 ON/OFF 具有完全相同模型参数量，但相比旧 D2h MLP Critic，新 GRU 使用了不同回归参数化、训练目标尺度处理与样本组织，**不能把 GRU 相较旧 MLP 的提高归因于历史**。
- 后续不应仅为追求某个时序模型更强而继续试探历史长度或反复筛选 epoch；价值预测问题仍可能主要来自回报尺度、未来回报随机性和不同阶段 on-policy 更新样本量。

## 正式准备工作方向

在与本诊断独立的 [官方 SB3-Contrib RecurrentPPO Smoke](2026-11-17-official-recurrentppo-smoke-results.md) 中，成熟 4-way MetaDrive PPO 链路已通过；16,384 步纯随机初始化 LSTM 仍 0/12 到达。因此本项目续训同一官方 checkpoint/VecNormalize 至约30万步的任务 [D2j](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44025266123) 用来验证长预算是否能建立驾驶能力，不作为 A/B/C 公平比较。等价值训练接口与训练系统明确后，再考虑 A/B/C 在严格配对预算下的 PPO 优化及 D3 消融，且不能以普通训练成功率替代因果模块协调证据。

## 已生成产物

- `/mnt/datasets/lxb/dls/reports/d2i-history/d2i-history-summary.json`
- `/mnt/datasets/lxb/dls/reports/d2i-history/history-ablation-{A,B,C}-seed42.json`
- `/mnt/datasets/lxb/dls/checkpoints/d2i-history/`

全程0个新增 MetaDrive 环境步、不覆盖任何旧 BC/PPO 权重。
