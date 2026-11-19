# DLS Research Instrument FRA v1 — Functional Responsibility Audit

日期：2026-11-19  
状态：**可复用审计组件已实现；数学真值正负对照通过，12 个受控学习模型实测完成；历史检查点只读复审通过。** 此文是测量仪器的记录，不声称新运动协调机制已经成立。

## 0. 为什么需要这件工具

先前 CRR 可学习输入拥有者和有向通信，但输入归属不等于真正的计算责任。D3b/D3d 表明仅检测边、活动变化或阻断消息带来的误差，不能证明任务的互补计算具体由哪些模块承担，也不能识别是否为真正的运动协调组。

工具 FRA 的任务是：
1. **函数层：** 输入变量 i、j 是否对模型输出形成非加性联合计算？
2. **模块层：** 哪些模块对该联合计算的输出贡献具有非零、可加总的作用？
3. **扰动层：** 阻断通信或禁用模块后，这些函数关系、各模块贡献、任务误差怎样变化？
4. **校准层：** 能否在预先已知的正例/阴性例中准确识别真实交互，同时暴露模型中错误的伪交互？

FRA **不是新的训练网络**，不为模块指定任务标签；先测量 CRR，再作为可复用探针迁移到驾驶系统。

## 1. 最简数学形式

对 F 维输入 x、可控制的替换样本 x' 以及变量 i,j，定义在模型**输出 logit 空间**中的混合反事实差分：

```math
I_{ij}(f;x,x')
=f(x)-f(x_{i\leftarrow x_i'})
-f(x_{j\leftarrow x_j'})
+f(x_{i,j\leftarrow x_i',x_j'}).
```

它对可加函数严格为零；对于乘积项通常非零。使用随机独立样本估计 RMS 交互强度。与二阶梯度不同，这种有限差分不要求导数存在，但结果依赖替换值和尺度。**独立替换只在变量独立的合成任务中属于可认可的反事实；真实相关车辆变量需要条件采样或物理一致的干预。**

对于当前 CRR 的线性输出头，logit 可准确写成

```math
\ell_o(x)=b_o+\sum_{k=1}^{K}c_{k,o}(x).
```

因此严格有

```math
I_{ij}(\ell)=\sum_{k=1}^{K}I_{ij}(c_k).
```

FRA 逐输入对、逐模块计算以上等式，并作为 fail-fast 数值校验。给定 **真值 logit** \(\ell^*\) 时，再计算模块对目标交互的带符号对齐：

```math
R_{k,ij}=\frac{\mathbb E\langle I_{ij}(c_k),I_{ij}(\ell^*)\rangle}
{\mathbb E\|I_{ij}(\ell^*)\|^2}.
```

归因可正可负，**不是概率**；由于各模块可以抵消，单项可小于0或大于1。

若真值未知，改用模型自身的 \(I_{ij}(\ell)\) 作为对齐参考，并验证各模块对齐系数的和约等于1。该评分解释当前网络的表示方式，**并非独立于网络基底的唯一责任分解，也不是因果物理结论**。

## 2. 代码与执行接口

- 核心库：`src/dls_functional_responsibility.py`
  - `CRRAdapter` 读取当前 CRR 的精确 per-module logit 贡献；
  - `audit_pairwise_responsibility(adapter, x, donor, target_logit=None, valid_interventions=True, ...)`
    对每对变量做四世界反事实、输出模型交互强度与各模块责任分量。无教师时 `target_logit=None`，可以选择提供 `target_output` 做损失审计。
  - `compare_failures(...)` 逐一禁用模块、自动重分配输入，并对比阻断通信的差异。
- 校准和训练脚本：`scripts/dls_functional_responsibility_experiment.py`；
- **独立的已保存 checkpoint 只读测量：** `scripts/dls_audit_responsibility_checkpoint.py`，支持 `--target none`（无需给归因器真值公式）以及 `--target oracle`；
- 工作流：`.github/workflows/dls-functional-responsibility.yml` 及 `.github/workflows/dls-responsibility-checkpoint-audit.yml`。

当前只读脚本内置与原合成实验一致的独立均匀样本生成器和标签发生器，用于检查 checkpoint 的完全一致性；对真实驾驶需要替换数据产生方式，不能直接拿它生成物理干预。

## 3. 精确函数校准

四变量任务，正例两路输出 logit 含 \(x_0x_1\) 和 \(x_2x_3\) 非线性乘积；阴性两路 logit 都是独立变量的线性相加。与输出 Sigmoid 的非线性区分开，所有交互测量统一在 logit 层。

- **阳性**：对数学真值，实现的真交互 (0,1)、(2,3) 被检测，其他四组交互严格近零；指定负责 (0,1) 的 Oracle 模块归因接近 1，另一真交互由另一模块承担；模块归因严格重建模型 logit 的二阶差分。
- **阴性**：6组混合差分均近零，且不产生可解释为任务真值的模块交互归因。
- **单模块失效**：按 CRR 提供的故障可用性门控重新路由，自动重新执行输出分解；所有恒等式在健康/失效条件分别验证。
- 3个数据/模型seed 7、8、9全部通过；原始训练/审计运行 [#44468230771](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44468230771)，SUCCESS。

## 4. 实测：12个 CRR 受控模型

4096 train / 1024 validation / 1024 test；4输入4模块，健康时每模块至多直接访问一个特征；linear 2-output logit readout；25 epoch Adam 0.007。4条件×3模型seed＝12个模型，checkpoint全部持久化。

| 条件 | 平均健康测试MSE | 平均单模块失效MSE |
|---|---:|---:|
| 互补函数，无通信 | 0.020177 | 0.022233 |
| 互补函数，有通信 | **0.000856** | 0.027087 |
| 互补函数，有通信 + 随机失效训练 | 0.002586 | **0.006653** |
| 可加函数，有通信 | 0.0000377 | 0.004392 |

在有通信互补函数的三个seed中：
- (0,1)目标交互对齐系数：0.956、0.864、0.885；
- (2,3)目标交互对齐系数：0.939、0.863、0.845；
- 跨模块贡献常分散在多个模块，而非精确按两两分组，每个seed的具体模块向量不同。

**值得保留的阴性发现：** 可加任务的真实 logit 混合差分全为零，但有通信学习器仍会形成小的非加性项。该模型健康测试MSE仅 0.0000377；阻断通信后 MSE 升到 0.007588。由此可见：**阻断通信损失很大不代表目标本身含互补交互。** FRA 可把模型内部的“伪交互”与任务真值结构分离报告。

**故障训练的结果：** 单模块失效平均误差由 0.02709 降至 0.00665，但健康误差 0.000856 升至 0.002586。这是准确度与抗扰性的权衡，**不是已证明剩余模块真正接管相同计算责任**：因为 CRR 故障后也自动改变输入分配，且当前没有“失效但不重分配”的严格对照。

运行报告：`/mnt/datasets/lxb/dls/reports/functional-responsibility-v1/aggregate.json`；每seed独立JSON和12份checkpoint保存在对应 reports/checkpoints 的同名版本目录。

## 5. 独立只读检查

复用 `interaction-comm1-fault0-seed7.pt` 冻结权重，分别执行：
- 不给测量器目标函数、只读取模型自身 logit 的交互；
- 再给数学Oracle，对相同变量对和相同反事实样本测量。

两套运行得到**完全一致的模型交互强度、模块自身投影与健康 MSE**；数学真值只改变是否能够进行目标结构正确性评分。工作流 [#44468813439](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44468813439)，SUCCESS。

结果路径：`.../reports/functional-responsibility-v1/probe-targetfree-seed7.json`、`probe-oracle-seed7.json`。

## 6. 研究工具的边界和下一步

**能回答：** 对限定的当前网络结构、明确的变量坐标、合法反事实样本，哪些变量联合影响输出 logit；这种联合计算由各模块的输出项怎样构成；阻断通信和禁用模块时责任图怎样改变。

**还不能回答：** 唯一确定生物式功能责任；从活动归因推断网络内部唯一因果机制；在多变量强相关的真实驾驶状态中任意交换输入并解释为真实环境干预；在有自动重分配输入的机制中直接宣称功能接管。

下一步顺序：
1. 增加**固定输入所有权不重配**的失效对照，与 CRR 现有可重配失效形成反事实差分，从而剥离“重新提供输入”本身的机械收益；
2. 添加模型层内消息/状态 patching，排除仅通过最终线性读出定位的归因局限；
3. 设计有效的**车辆动力学条件反事实发生器**，将 FRA 接入 DLS 的连续闭环轨迹，而非继续只使用合成标量计算。

相关成熟技术参考：functional ANOVA / 交互有限差分、Causal Abstraction（Geiger et al., 2023），activation patching/causal scrubbing；FRA 是针对 DLS 研究约束的最小组合式测量工具，不宣称上述思想为全新算法。参见 https://arxiv.org/abs/2301.04709 与 https://openreview.net/pdf?id=zSf8PJyQb2 。

**科学结论：测量器已校准可用；自主计算责任的形成与真正动态接管仍是下一阶段需要研究的现象。**
