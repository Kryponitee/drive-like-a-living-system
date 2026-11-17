# D2l 完整 PPO 训练性能验收结果

日期：2026-11-17。原始、优先使用的 [官方 D2l 工作流 #44030181386](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44030181386) — **SUCCESS**。重复排队的 16k benchmark 已取消；只读复核 [#44030454226](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44030454226) 成功。没有再进行第二遍 16k 训练。

## 科学/工程目的

D2k 在固定 LSTM actor、无反向传播的采样测试中 64 workers 优于32/48。但它不能反映完整 PPO 的瓶颈。本次从同一个 SB3-Contrib 2.6.0 / PyTorch 2.5.1 官方 D2j 300k checkpoint 和同一 VecNormalize 统计**重新加载**两个实验条件，严格复用原训练框架，比较整段 MetaDrive 采样 + 递归策略 PPO 更新的真实 wall clock。

- 配置 32 与64 workers，依次运行（不同时争用），各 **32,768 个实际环境转移**，地图3、密度0.1，headless MetaDrive；
- n_steps/worker=128，minibatch=128，n_epochs=4，模型、奖励规范、归一化完全保持；每配置 **1,024 次 optimizer.step()**，已经在日志与 JSON 中验证；
- 不保存新的 actor 权重，未覆盖 D2j checkpoint、VecNormalize 或旧 D1/A/B/C 模型。
- 注意：32 workers 一次采样4096帧 ×8个 rollout，64 workers 一次8192帧 ×4个rollout；**PPO 策略更新频率不同**，因此本项只能评估工程吞吐，不能推断相同的学习动力学。

## 实测

| workers | 完整训练 FPS | 学习墙钟秒数 | 采样及框架开销秒数 | PPO 训练函数秒数 | optimizer steps |
|---:|---:|---:|---:|---:|---:|
| 32 | **224.01** | 146.28 | 60.79 | 85.49 | 1024 |
| **64** | **258.19** | **126.92** | **42.11** | 84.80 | 1024 |

- 完整 PPO 64/32 FPS 比 = **1.15259×**，加速 **15.26%**，比预设的至少10%门槛略高。
- 随着 CPU workers 增多，采样/相关框架开销缩短 `60.79→42.11` 秒；PPO train() 时间 `85.49→84.80` 秒几乎未变。训练计算已占总耗时更多部分，继续增 workers 预期边际收益更小。
- 32/64 测试都在**CPU**执行策略网络更新；后续的 A/B/C 复杂网络可以验证 GPU 训练收益，但不能未经测量就宣称 GPU 一定更快。

## 决策与准备工作收口

**固定 64 workers 作为下一阶段默认多环境采样数。** 不再测96/128 workers，不再以 worker tuning 代替主科学实验。

下一步 D2m 是**唯一必要的自定义状态正确性检查**：SB3-Contrib 2.6.0 的 `RecurrentPPO` 根据 `lstm_actor.num_layers, lstm_actor.hidden_size` 构造回放缓冲区，并调用 `forward/evaluate_actions/get_distribution/predict_values`，不能把现有 A/B/C `step(obs,(h,z),scalar_t)` 直接交给它。不同 MetaDrive worker 会异步 reset，而 A/B/C 慢环每5步更新，因此每 worker 必须独立维护 `clock mod5`、h、z。详见 [官方源代码接口记录](2026-11-17-official-RL-stack-audit.md)。

D2m 通过后，必须把状态接入**成熟 PPO 采样/优化接口**（或由官方框架能力支持的标准更新组件），而不新写一份完整的 PPO；再做 D2n **BC 初始化的 A/B/C 正式训练与后续 D3 模块干预**。不需要再为随机初始化普通 LSTM 投入长预算。

数据盘文件：`/mnt/datasets/lxb/dls/reports/d2l/full-ppo-worker-comparison.json`，及 `full-ppo-32-workers.json`、`full-ppo-64-workers.json`。
