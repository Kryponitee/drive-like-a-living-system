# DLS — 官方 SB3-Contrib RecurrentPPO 训练系统基线（独立烟测）

日期：2026-11-17。最终 [GitHub Actions #44024379218](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44024379218)：**SUCCESS**。

## 为什么做

D2b–D2g 使用自写 recurrent PPO、4096 个环境步，没有长期收敛证据。为了区分通用 PPO 工程问题与 A/B/C 模块协调的科学问题，先用成熟的 **SB3-Contrib RecurrentPPO** 在同一 MetaDrive 0.4.3 环境上独立完成训练系统 smoke。此参考基线**从随机初始化的标准 LSTM 开始**，和 BC 初始化、不同结构的 A/B/C 不构成公平的架构对照。

## 官方实现和环境隔离

- 核查了官方 MetaDrive [Training](https://metadrive-simulator.readthedocs.io/en/latest/training.html) 示例的子进程环境训练方案、[SB3-Contrib RecurrentPPO 2.6](https://sb3-contrib.readthedocs.io/en/v2.6.0/modules/ppo_recurrent.html) 的 LSTM/episode_start 约定及 [SB3 2.6 依赖](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/setup.py)。
- 最新 SB3 2.9.2 要求 torch≥2.8；项目核心 DLS 使用 torch2.5.1，Gymnasium1.4.0。选官方 **SB3=2.6.0, sb3-contrib=2.6.0, Gymnasium=1.1.1**，保护原 torch2.5.1、MetaDrive0.4.3 和 NumPy。
- 单独建立根盘轻量共享包环境 `/home/ps/miniconda3/envs/dls-sb3-compat`（`venv --system-site-packages`），复用 `dls-p0` 的 torch/Panda3D/MetaDrive，不复制原有权重/资源；`pip --dry-run`、`pip check` 与导入全过。检查证实原有 `/home/ps/miniconda3/envs/dls-p0` 的 Gymnasium 仍为1.4.0，没安装额外 SB3 包。
- MetaDrive 要求 reset seed 在 `start_seed` 配置区间内；SB3 给 VecEnv 传独立的全局随机种子。仅在接入适配层内将 seed 映射到每个 worker 私有的有效区间；不改 MetaDrive 源码/物理动力学。该适配层必须作为实验代码保留。
- MetaDrive0.4.3 reset 不接受 Gymnasium 的 `options` 参数；适配层只允许 None/空字典并向原 API 传递 `seed`。所有接口修复已通过真实多进程运行验证。

## 完成的训练测试

- 官方 `RecurrentPPO(MlpLstmPolicy)`，4 个 `SubprocVecEnv`，每个 worker 独立 start_seed 区间 `[5000,5080),[5080,5160),[5160,5240),[5240,5320)`。
- `VecMonitor` 和 `VecNormalize`：训练时观测/奖励归一化，评测时固定观测统计且使用环境原始奖励。
- MetaDrive structured state259、动作Box(2)、map3、traffic_density0.1、horizon800。
- `n_steps=128`（每环境）、`batch_size=128`、`n_epochs=4`、`gamma=0.99`、`lambda=0.95`、`log_std_init=-3.0`（动作采样约sigma0.05）、CPU 推理/优化、LSTM hidden64。
- **16,384 个真实环境转移**，全部训练并保存 checkpoint、VecNormalize stats 和 TensorBoard。训练 75.6 秒，末次输出统计的整体吞吐约218环境步/秒。
- 标准 SB3 训练日志末次 `explained_variance=0.593`（基于其训练价值目标和归一化统计，**不与 D2g 原始回报 EV 直接比较**）。
- 新场景 `5400..5411`，完全隔离于训练区间，确定性闭环**0/12 到达、1/12 碰撞、10/12 越界**；平均原始 episode reward `73.17`。这只是**16k-step smoke**，不能判断长期学习是否可达，更不能与已用专家初始化的 A/B/C 排名。

## 科学结论与精修

**正面**：官方开源算法能在我们的 MetaDrive、Panda3D 和资源隔离条件下正常完成并行采样、递归状态管理、PPO 更新、指标输出与冻结模型评估。避免继续自写这些成熟算法的标准部分。

**未得到**：官方随机 LSTM 在16k步下学会可靠驾驶；与 A/B/C 的统一容量、初始能力和训练预算比较；更没有模块间通信因果证据。官方示例使用的30万步是训练时长的参考数量级而非保证或严格预算。

**下一步**：
1. 独立运行已预注册的 [D2i-H 历史信息 ON/OFF](2026-11-17-D2i-critic-history-control-plan.md)，用现有 D2h 数据确定时间上下文对 Critic 的作用，完全不消耗新环境步数。
2. 只有在解释力/训练机制明确后，才将成熟的并行采样、GAE、PPO 价值归一化等部分适配到 A/B/C 的自定义循环状态。SB3 原生 LSTM 不可直接替换四模块 GRU+慢环架构；如采用 TorchRL，应单独验证时间时钟、reset、动作分布和旧策略重放。
3. 正式 D2 需要足量环境步（例如官方示例30万作为初步量级）及多个初始化训练种子；必须固定并记录训练采样开销与状态分布。D3 才评估通信扰动因果性。

## 产物

- `/mnt/datasets/lxb/dls/reports/d2i/official-rl-smoke.json`
- `/mnt/datasets/lxb/dls/reports/d2i/official-pip-dry-run.json`
- `/mnt/datasets/lxb/dls/checkpoints/d2i/official-recurrentppo-16k.zip`
- `/mnt/datasets/lxb/dls/checkpoints/d2i/official-recurrentppo-vecnormalize-16k.pkl`
- `/mnt/datasets/lxb/dls/tensorboard/d2i/`

所有模型、数据和实验报告均在**独立数据盘**；root disk 仅存隔离轻量兼容环境，无核心环境变更。
