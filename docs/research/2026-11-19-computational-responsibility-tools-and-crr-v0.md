# DLS 研究记录：计算责任是否有现成机制？CRR 原型与最小检验

日期：2026-11-19  
状态：**文献对照完成，CRR v0 可运行、接口检查通过；仅玩具任务的早期正向信号，不是运动协调组形成的证明。**

## 研究问题

给定整体运动控制函数 y=f(x)，一组具有局部容量上限的模块能否**在不预先赋予物理变量角色或正确配对标签**的前提下，学会分配输入访问、选择通信对象、联合计算，并在模块暂时失效时调整输入访问与计算？

严格区分：输入归属 G、网络通信 A、实际功能责任 R。G 和 A 是训练出来的结构变量；R 只能通过任务相关的因果干预与损失变化核验，不能通过 G、A 本身断言。

## 已有方法与缺口

- RIM（Goyal et al., ICLR 2021）：模块激活竞争、近独立循环动力学、稀疏通信；能支持模块专业化，但并未解决 DLS 所要求的具体功能责任归因和失效后的任务级接管。论文 https://mlanthology.org/iclr/2021/goyal2021iclr-recurrent/ ，作者实现 https://github.com/anirudh9119/RIMs 。
- Soft Modularization（Yang et al., NeurIPS 2020）：学习任务条件的模块路由和复用；但原论文任务本身已具备多任务监督，不等于无语义标签的运动自由分工。https://rchalyang.github.io/SoftModule/ 。
- TarMAC（Das et al., ICML 2019）：学发送什么、发给谁；在合作式多智能体任务中有效，但不是神经子模块计算责任的因果归因方案。https://proceedings.mlr.press/v97/das19a.html 。
- Béna & Goodman, Nature Communications 2025：结构模块化不能保证功能专业化；具有意义的可分离环境因素、强资源限制、输出瓶颈与信息流带宽影响分工。https://pmc.ncbi.nlm.nih.gov/articles/PMC11695987/ 。
- Mittal et al., NeurIPS 2022：现有端到端学习的模块架构无法总是发挥已知模块解的潜力。https://proceedings.neurips.cc/paper_files/paper/2022/hash/b8d1d741f137d9b6ac4f3c1683791e4a-Abstract-Conference.html 。
- Wu et al., Nature Machine Intelligence 2026：多任务与容量压力下 RNN 形成任务相关结构模块化；官方复现 https://github.com/brain-intelligence-lab/modular-brain-rnn ，但未直接覆盖 DLS 的执行器反馈与单模块失效接管。
- Willems et al., arXiv 2026-11-10：SyncNet 通过动态相位通信研究功能模块；作者也指出模块究竟计算什么、是否跨任务复用未得到验证。https://arxiv.org/abs/2611.13759 。

结论：**不需要重新发明 RIM、注意力或二部匹配；需要自己的任务相关责任验证工具与最小组合原型。** 这不构成方法新颖性主张。

## CRR v0：最小工具（非正式命名）

源码：`src/dls_computational_responsibility.py`；测试：`scripts/dls_crr_prototype_smoke.py`。

设有 F 个输入变量、K 个模块。输入归属矩阵 G 属于 {0,1}^{K×F}，每列恰有一个拥有者。可选的均衡资源限制在当前健康条件下要求每模块至多 ceil(F/K) 个变量，使用 Sinkhorn 连续松弛、Hungarian 整数匹配和直通梯度估计。这里的约束是**架构给定的容量偏置**，不是已学出的生物组织。

```text
x  -- learned feature ownership G --> local encoder h_i
h_i -- targeted attention A_ij --> receive m_i
(h_i, m_i) -- private nonlinear receiver --> scalar activity a_i
[a_1,...,a_K] -- LINEAR readout --> 2-channel sigmoid output
```

输入归属与通信权重都没有使用人工功能标签。模块失效时屏蔽其输入、收发消息和输出，剩余输入重新分配给存活模块，允许剩余模块的有效输入预算相应增加。**故障期间改变输入访问是机制提供的，不代表控制功能被成功接管；需要额外验证。**

接口导出 feature_ownership、directed_attention、module_activity、module_logits，支持逐模块缺席造成的任务损失变化，但尚未支持完整的反事实变量×模块责任张量。

## 任务和可复现的原型试验

原型采用 x0,x1,x2,x3 独立均匀分布在 [-1,1]；目标是两个 Sigmoid 输出，其 logits 均包含 x0*x1 与 x2*x3 的非线性联合项。训练 4096 样本，测试 1024，四个同构模块，健康时每模块只允许访问一个输入变量，linear readout；25 epoch、Adam lr=0.007、batch256。没有驾驶物理环境、没有高层 N、没有序列时域反馈。

GitHub Actions [#44466202879](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44466202879) **SUCCESS**；PyTorch 2.5.1+cu124, SciPy 1.15.3。完整结果：`/mnt/datasets/lxb/dls/reports/crr-prototype-v0/run-44466202879.json`。

| 训练 seed | 无直接通信（健康测试MSE） | 允许定向通信（健康测试MSE） |
|---|---:|---:|
| 7 | 0.0199703 | **0.0000506** |
| 8 | 0.0204250 | **0.0015478** |

两种通信结构的训练预算相同，输出头同为线性；但**在每模块只能访问一个变量的容量约束下，没有跨模块通信就无法直接计算 x0*x1 等交互项，所以这是人为设定的阳性功能校准，不是发现“通信本质上优于一般单体网络”。**

额外的失效试验（seed=7，通信启用）：

| 条件 | 健康测试MSE | 四种单模块失效的平均测试MSE |
|---|---:|---:|
| 只训练正常情况 | **0.0000506** | 0.0480667 |
| 每训练batch以40%概率随机失活一个模块 | 0.0031796 | **0.0074666** |

含义：容量限制会产生对通信的依赖；随机失效训练能显著改变模型的容错权衡，但也损害健康场景的拟合精度。**只测试了一个故障训练seed，不能声称获得稳定、可复用的责任接管；重路由也不等于真正功能补偿。**

未限制容量时，本地先导试验中出现过同一模块接受所有四个输入的退化解，提示若无客观资源压力，网络可把计算集中处理，根本无需模块协作。

## 边界和下一步

1. **责任归因：** 增加变量级反事实变化与模块内部激活patch，报告谁实际上在计算目标间交互，而不是仅检查 G 和 A；在可加阴性任务中应不出现虚假的交互责任。
2. **自发性：** 将固定均衡容量（预先限制单模块输入数）与不限容量、通信带宽、有限模块状态容量分别消融；避免把人为分配约束误称为完全自然的分工。
3. **责任重分配：** 同时检查正常与单模块失活，训练更多seed及控制同等模型参数/训练预算的单体网络；目标是验证失效后控制功能保持程度，而非纯粹改换拥有者。
4. **回归 DLS：** 仅在上述工具通过校准后，接入 P0 纵向动态反馈或更丰富的车辆运动任务，研究 N/C 分层控制与实际运动稳定性。静态函数拟合不能替代闭环运动协同证据。

科学门槛：若某方法只能让模块看不同变量、通信并降低损失，而无法反事实说明谁计算什么，以及这种责任能否在干扰后恢复，**它仍不是成功的运动协调组模型**。
