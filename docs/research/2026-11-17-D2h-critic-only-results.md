# DLS — D2h Critic 独立训练验证结果与方法边界

**日期：2026-11-17。执行记录 [GitHub Actions #44023029712](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44023029712)，完整 PASS。**

## 原始目的

D2g 测得现有 D2d/D2f Critic 对实际未来折扣回报的解释方差约 0.8%–1.9%。D2h 为确定「Critic 是否至少能够通过标准监督学习捕捉回报变化」，冻结 BC actor，仅在 MetaDrive 全 episode 的实际状态 / `γ=0.99` 折扣回报上对**同一个原 Critic 结构**运行 PyTorch MSE 回归，不更新策略、不运行 PPO。

实验共 6 条独立采集作业（3 actor × train/val）和 3 个 GPU 的 Critic 监督训练。所有资源继续复用系统盘 Conda、数据盘资产。

- 训练场景 seeds `4600..4623`（24 episode / actor）；
- 验证场景 seeds `4700..4715`（16 episode / actor）；
- 同一 BC 训练初始化 seed42、地图3、密度0.10、tanh 前采样标准差0.05、最大800步；
- 只对**实际终止的完整 episode**使用 Monte Carlo Return 监督，触及 horizon 的未完成 episode 整条标记为删失并排除，防止将人工截断误当为真实未来回报零值；
- Critic 初始化自各自 D2f checkpoint（同结构 `259→64→64→1`），固定50个 epoch、batch512、AdamW lr0.001，原始回报不归一化；
- 独立验证集比较：旧 D2f Critic、训练后 Critic、训练集回报常数均值基线；没有用验证集选择 epoch 或超参数。

## 原始结果

| 模型 | 训练有效步 | 验证有效步 | 原 Critic EV | 新 Critic EV | 原 Critic RMSE | 新 Critic RMSE | 常数基线 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 7,918 | 5,723 | 0.0228 | **0.2145** | 63.980 | **22.791** | 23.767 |
| B | 8,835 | 6,486 | 0.0109 | **0.1617** | 65.509 | **23.764** | 24.750 |
| C | 8,297 | 5,867 | 0.0135 | **−0.1548** | 65.208 | 30.067 | **25.176** |

注意：`EV=1-Var(G-V)/Var(G)`，不是准确率。值越高越能解释回报波动；负值表示残差方差比目标方差更大。RMSE 越低越好。每个 actor 的采样策略不同，访问信息分布有差异；不得将不同 actor 的 EV 当成同样测试分布下的公平架构比较。

A/B 在独立场景中得到非平凡回报拟合提升，且 RMSE 略低于常数基线；C 虽然比原有近常数估值 RMSE 大幅降低，但严重低于常数基线，存在泛化不良。故“Critic 无法表示未来回报”这一绝对说法不成立，“简单加长 Critic 训练就能解决 PPO”也不成立。

## 对 D2 准备工作的精修

**当前已经确证：**

1. D2b–D2c 的 sigma0.27 让 BC 驾驶成功几乎消失，属于明显的动作探索匹配问题；
2. D2f BC 功能锚定能降低离线函数漂移，但闭环改善不统一；
3. D2g 的 Critic 估值质量很低；
4. D2h 标准回归能够部分提高价值拟合，C 的独立验证却退化至不如常数基线。

**当前尚不能确证：** 价值拟合不好是 PPO 更新失败的唯一原因；更不能将这些工程问题解释为运动模块不能形成协同。

在正式继续 PPO 前，应使用成熟库建立独立系统对照，并审计 reward / observation normalization、recurrent hidden-state reset、time-limit bootstrap、value loss 等实现细节。参考 [官方 RL 技术栈审计](2026-11-17-official-RL-stack-audit.md)。

## 实施兼容性检查

D2h 的只读环境包检查实测：
- Python 3.10、PyTorch 2.5.1，Gymnasium 1.4.0；`stable-baselines3`、`sb3-contrib`、`torchrl`、`tensordict` **均未安装**。
- 最新 SB3 2.9.2 要求 PyTorch ≥2.8；安装它会改变既有依赖，不应直接在 `dls-p0` 中 pip install。
- 官方较早的 SB3/SB3-Contrib 2.6.0 要求 `torch>=2.3,<3.0` 且 `gymnasium>=0.29.1,<1.2.0`。因此本项目选 **PyTorch 2.5.1 继承 + 独立 root-disk 轻量 venv 覆盖 Gymnasium 1.1.1** 的方式进行小规模官方 RecurrentPPO 系统验证，而不是降级/升级主环境依赖。
- SB3 的标准 MlpLstmPolicy 仅用于检查官方 PPO 训练链路是否有效，既不是 D1 BC 初始化，也不是 A/B/C 的容量/架构匹配消融。

官方实施工作流：[D2i](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44024068698)。任何官方 LSTM 的初步结果都不得拿来宣称或否定 A/B/C 的模块通信收益。

## 研究路线

- **D2i**：锁版本、小环境隔离、官方 RecurrentPPO 4 环境并行 + 16,384-step 训练链路烟测；检验安装/重置/GAE/归一化/评价可用性，而不是在16k步上宣称收敛。
- **D2j**：在 D2i 正常通过后才评估是否有必要让官方基线训练到足够环境步数，并先检查 MetaDrive 对中等密度随机 LSTM 的学习难度。
- **D2k**：面向 A/B/C 的成熟 PPO 适配与同等训练预算对照，重点是自定义循环状态与慢环时钟，不允许用官方 LSTM 替换科学变量。
- **D3**：完成基础控制对照条件后，冻结 C 模型做通信边干预 / B 的局部扰动对照，以此判断运动模块的因果作用。
