# D2n-A — BC 初始化的正式 A/B/C 闭环 PPO 首轮对照（预注册）

日期：2026-11-17。D2l **64 workers optimizer-inclusive** 验收成功；D2m 异步时钟 / D2m+ 官方 RecurrentPPO 自定义策略接入均 SUCCESS。准备工作据此正式收口。本轮是**实际科研对照的第一轮**，而不是继续泛用 PPO 工程验证。

## 唯一核心问题

**在完全相同训练预算与已有驾驶能力起点下，官方闭环优化能否增强 A/B/C 的 MetaDrive 控制性能，而不是破坏既有 BC 能力？B/C 的差异是否给下一阶段消息因果干预提供候选信号？**

## 固定协议

- 模型：原有 `src/dls_p0.py` A/B/C、原 D1 `seed42` BC checkpoint。非官方随机初始化 LSTM；64-worker 的 D2j 原 checkpoint **只用于此前训练系统验收**，不用于本实验。
- A/B/C 独立训练，统一 `seed=811`、相同 64 个独立 MetaDrive CPU worker 场景配置，密度 0.10，地图 3，起始场景 `start_seed=24000+80*i, num_scenarios=80`，故总训练地图池 24000–29119。
- 每模型 **65,536 个环境交互步**，即8个 `64×128` rollouts；`batch_size=512`、`n_epochs=4`、`gamma=0.99`、`GAE lambda=0.95`、`clip_range=0.2`、`lr=5e-5`、`vf_coef=0.5`、`ent_coef=0`、`max_grad_norm=0.5`。所有超参数对三个网络完全一致；短预算仅作探索性一轮，不宣称收敛。
- 初始化三者的 `log_sigma=log(0.05)`；使用成熟 SB3-Contrib 2.6.0 官方 PPO collector/recurrent rollout buffer/GAE/loss/optimizer，保留 A/B/C 原始通信与每环境独立 slow clock，不重新实现 PPO。
- `VecNormalize` **只归一化训练 reward，不额外归一化 observation**，因为自定义 policy 已用 D1 成功示范数据标准化输入；Critic 三组同一 `IdenticalCritic` 结构、独立 RNG fork 固定初始化。`BC` 初始权重与标准化完全保留。禁用新专家数据。
- A/B/C 训练**顺序使用64 workers**，避免超配服务器128逻辑线程；现有 GPU 不在这一 CPU 短试验的性能断言范围内。所有训练权重、VecNormalize stats、TensorBoard、日志写入数据盘，不覆盖 D1/D2 权重。
- PPO 更新前通过官方 buffer 采样检查动作 logprob ratio≈1，若重放误差超过2e-3则 fail fast；每模型完整执行8个rollout、65,536 env steps。
- **主评估**：用完全未出现过的 scene seeds `30000..30023`，三种密度0/0.1/0.25，每个模型相同场景配对测原 BC 与新 PPO 的 deterministic success、collisions、out-of-road、route completion、平均原始环境奖励。B/C 对比基于相同场景的成对差异，A 是外部参照。
- **能力保留**：另在 D1 9822 步独立验证集上比较 PPO actor 与专家动作的 teacher-forced Huber；如果昂贵则先只读回归，绝不拿训练数据做能力保留结论。
- **判据**：以密度0.1 的全新场景 successful completions 和平均原始奖励为主；只有BC能力没有明显丢失、B/C趋势在新场景可重复，才扩大训练seed 43/44和最终预算。即便 C 高于 B，仍不等于有模块因果协调；D3 的消息切断/局部扰动才检验主假设。

## 接下来两步

**D2n-B**：在首轮结构与控制能力成立时，对训练初始化42/43/44做相同协议重复，依据 C-B 的 paired success 和 reward，以及 A 的整体表现决定是否需要加预算。禁止事后挑有利密度或地图。

**D3-A**：冻结 C 的策略直接做消息屏蔽、延迟与同量级 B 局部扰动，对任务损失和恢复曲线建立因果证据。若 PPO 仍破坏 BC 能力，应回退至 D1 BC 冻结策略做 D3 可行性测试，不把 PPO 工程瓶颈无限延长为科学前置条件。

## 重要约束

此轮64 workers为**已验收的系统性能配置**，不能据此假设 A/B/C 自定义控制器具有与官方简单 LSTM 相同吞吐。A/B/C 本体没有变，B/C 模型训练参数量相等但推理 FLOPs 不一定相等；不完成计算成本审计不得归因于通信机制独立优越性。
