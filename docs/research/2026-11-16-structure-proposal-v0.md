# DLS 设计草案 002：从连接学习到运动协调结构（待审查）

- 日期：2026-11-16
- 状态：**提案，未获批准、未实现、未做实验**
- 接续：[`001` 最小定义与文献](2026-11-16-motor-coordination-definition.md)

## 一句话设计

**不预设 8 个运动控制器，也不预设「转弯/制动/跟车」模块。先搭建一个共享的有状态神经运动系统，让其神经连接在驾驶学习中被优化；学习后再检验是否自然形成可复用的功能群落。**

强调：**学到了神经连接结构 ≠ 学到了运动协调结构**，后者必须通过任务反馈与因果干预来验证。

## 最小系统接口

- 高层（慢环）：观察场景、导航目标、执行反馈，产生协调条件 `z_k = N_theta(o_k, g_k, f_k)`。
- 底层（快环）：一个有状态神经动力系统 `C_phi` 接收车辆实时状态 `s_t` 和条件 `z_k`，输出控制建议 `u_t`；每一步都读取新状态。
- 车辆执行器与动力学：真实/仿真执行、返回状态。安全控制单独处理（不宣称学习可提供安全保证）。

选择「一个整体可学习网络」而不是「先划分多个独立神经网络模块」，以避免手工指定模块边界或数量。这里的「神经元」是计算单元/隐状态维度，并非生物运动神经元或真实肌肉。网络总容量仍需工程设定。

### 首选候选：可学习稀疏循环连接

```text
场景/导航/反馈 -> 高层 N -> 条件 z
                               |
车辆实时状态 ---------------> 共享循环神经网络 C(稀疏连接可学习) -> 控制输出
       ^                                                    |
       +------------------- 车辆执行与反馈 <------------------+
```

可写作（结构提案，非既定实现）：

```math
h_{t+1} = sigma(W_in [s_t, z_k] + (M ⊙ W_rec) h_t + b)
u_t = D(h_{t+1})
```

其中 `M` 是连接掩码，`W_rec` 是可训练权重。训练中可采用 RigL 风格的低幅度连接删除与梯度驱动重新生长；**先不允许每步生成新的网络拓扑**。在推理中拓扑可固定，神经活动仍在反馈下持续变化。

### 学什么，不学什么

| 学习对象 | 是否自主学习 |
| --- | --- |
| 各神经连接权重 | 是 |
| 哪些循环连接保留/出现 | 是（动态稀疏训练候选） |
| 各单元的活动和随反馈变化 | 是 |
| 是否形成可重复的功能群落 | **不假设**，训练后测量 |
| 神经元总预算、输入输出接口、反馈回路 | 工程设定 |
| 具体「转弯/刹车/跟车」职责 | **不预设** |
| 模块数量和边界 | **不预设**；如出现可用社区发现提取，但提取结果非唯一 |

不把社区发现算法识别出的连接簇直接称为运动协同。验证单元间的反馈补偿和跨场景复用，是后续必要条件。

## 学习与判定：最小路径

1. **离线学习运动**：先学习专家的状态-动作和动作后状态变化，不赋予内部功能标签。此阶段最多证明可以控制，不能证明协同。
2. **闭环反馈优化**：让动作影响后续观测，在受控干扰和多样任务条件下训练；是否使用强化学习需进一步选定。
3. **结构测量**：分析训练后的连接图、隐状态协同动态是否稳定。比较不同随机种子、环境和训练阶段；社区编号不要求逐一相同。
4. **功能干预**：破坏被发现的连接/局部动态，与等强度随机破坏比较，检验反馈误差和任务变量稳定性。
5. **复用与价值**：变化速度、道路几何与交互场景，比较整体网络、固定随机稀疏网络、动态稀疏网络在闭环成功、恢复时间和泛化上的表现。需匹配模型容量、数据、训练与推理预算，不能仅凭更稀疏得出更强的结论。

**停止/否定条件**：
- 仅学到稀疏网络但没有稳定功能群落：不宣称涌现运动协调。
- 存在群落但干预后没有对应的功能影响：不能称为有效协调。
- 干预有影响但无反馈适应/跨条件复用：只能称为内部功能组织，不能主张我们的关键协调优势。
- 普通 RNN 有同样性质/效果：不额外主张结构学习带来了收益。

## 为什么此方案适合作为 DLS-v0 建模起点

- 比「预设若干专家/技能」更少人为语义假设；
- 比直接放弃结构性、仅训练黑箱网络多了一个**可学习连接图**；
- 与 NRI 的思想（从动态观测推断关系）相关，但此处的节点是神经隐状态，不是 NRI 中预先给定的物理对象；
- 与 RigL 的训练拓扑改变方法一致，但 **RigL 研究了稀疏网络优化，不保证功能模块形成**。

## 相关文献（支持范围与边界）

- Evci et al., *Rigging the Lottery: Making All Tickets Winners*, ICML 2020: 稀疏拓扑训练中删除低权重连接并用梯度增加连接。https://proceedings.mlr.press/v119/evci20a.html
- Kipf et al., *Neural Relational Inference for Interacting Systems*, ICML 2018: 在已知实体的前提下联合学习交互图和系统动态；不直接等于学习神经元群落。https://proceedings.mlr.press/v80/kipf18a.html
- Béna & Goodman, *Dynamics of specialization in neural modules under resource constraints*, Nature Communications, 2025: 结构模块化不保证功能专门化。https://www.nature.com/articles/s41467-024-55188-9
- Wu et al., *Task-structured modularity emerges in artificial networks and aligns with brain architecture*, Nature Machine Intelligence, 2026: 多任务/增量任务训练在受限 RNN 中能促进结构模块化，尚不能直接外推至驾驶。https://www.nature.com/articles/s42256-026-01306-9

## 审查问题（只需要审这一点）

**是否允许以「单个循环网络 + 可学习稀疏连接 + 训练后发现功能群落」作为第一版协调系统的建模假设？**

如不同意，再对比「连续潜在动力系统（完全不做连接学习）」或「预设模块容量但学习门控/交互」两种路线。

本文件只提交研究提案，不进入模型实现或实验。


## Review update — 2026-11-16

User **accepted v0 as a deliberately loose first-pass structure hypothesis**. This approves only the research direction (learned coordination organization instead of hand-labeled motor units). Specific topology, sparse training method, model, platform, and experiments remain undecided. A staged perception interface is proposed in [record 003](2026-11-16-perception-integration-v0.md) and awaits review.
