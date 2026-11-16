# DLS P0 — 环境与数据规模预算

日期：2026-11-16  
状态：**研究工程预算（估算）；尚未安装 MetaDrive，也未进行资源实测**。  
对应方案：[P0 v1.0](2026-11-16-P0-protocol-v1.md)。

## 最小环境

- 平台：Linux（Manjaro/Ubuntu 等）；Python 3.10/3.11 作为保守兼容起点。请锁定版本和依赖，先单机验证；不把具体发行版视为官方最低要求。
- 软件：MetaDrive（从官方 Git 仓库安装并固定 commit）、PyTorch、NumPy、Gymnasium；后续采用能够正确处理 recurrent hidden state 的 BC/PPO 训练代码。
- 硬件：CPU 8–16 核更有利于并行模拟，RAM 推荐 32GB（16GB 可先尝试），GPU 推荐 ≥12GB VRAM，NVMe 预留 30–50GB。
- **P0 仿真不需要图像采集**：MetaDrive 默认 LidarStateObservation 可提供自车、导航和周围环境的向量信息；射线式传感器不依赖实时 GPU 图像渲染。P0 GPU 用于 PyTorch 模型训练，不强制用高端图形卡模拟摄像头。
- MetaDrive 文档指出同一进程仅运行一个环境实例，扩大采样吞吐使用多进程而不是多线程共享环境。
- 不承诺特定 FPS 或训练时间，吞吐受 CPU、交通密度、地图配置及策略批处理影响。

## 原始数据来源

优先使用 MetaDrive 程序化道路/交通生成器和内置 ExpertPolicy/IDMPolicy 采样专家交互（须先验证这些策略在所选场景下的成功率与行为多样性）。
不需要下载大规模真实世界图像集或 NAVSIM 数据。
每条示范记录可以保存 `(obs_t, action_t, obs_{t+1}, done, scenario_seed)` 和必要的任务反馈；按 episode 保存序列次序、控制器重置点。动作是连续 `[steer, throttle_brake]`。
禁止保存/输入测试 seed 生成的专家训练轨迹；策略不能读取未来轨迹或仿真器内部答案。

## 数据量及资源估算（非实测）

| 阶段 | 初始预算 | 持久存储 |
|---|---|---|
| D0 | 1千–5千步功能审计，覆盖若干地图及交通条件 | 几 MB |
| D1 | 10万–50万步连续专家示范（先小样本审计有效覆盖） | 约 0.2–1.2GB（理想紧凑 float32 容器，视维度/元数据而变） |
| D2 | 100万–500万环境交互步作为探索性训练预算 | 不强制逐步永久保存；rollout 在内存及按需日志 |
| D3 | 100–200 个独立测试场景作为首轮评估，按场景与训练 seed 配对 | 指标和抽样轨迹为主，可按需扩展 |

### 存储量公式

若单条示范保存当前与下一时刻 **256 维 float32** 向量、2维 float32动作，则纯数组字节数近似：
```
bytes_per_transition = (256 + 256 + 2) * 4 = 2056 bytes
100000 transitions ≈ 0.206 GB (decimal)
500000 transitions ≈ 1.028 GB (decimal)
```
元数据/episode 索引、其他状态和文件格式可能增加占用；这不是 MetaDrive 官方承诺的观测维度。若采用数组压缩或仅存观测+动作，数字会下降；若使用对象序列化或保存渲染数据，可能显著上升。

初步 P0 控制网络只有 K=4 个 d=32 的循环模块，加轻量 N/路由/共享控制头，预期远小于视觉模型；实际参数量应由实现打印，不把工程初值当作已测算网络规模。
训练 checkpoint、实验重复与日志使用 30–50GB 总盘空间更稳妥，但这属于预留预算而非实际文件要求。

## D0 必做验证

1. 固定 MetaDrive 版本/commit，核验 Python、PyTorch、Gymnasium 兼容及无显示设备运行方式。
2. 记录实际观测维度、每步环境耗时、RAM/CPU 占用；训练前做数据泄漏审计。
3. 先跑内置专家不同场景的表现，确认可提供真实有用的连续控制示范及必要的反馈序列。
4. 测试可复现地重置与复现受控干预；检查所需场景是否真实支持。
5. 再据实测观测维度和每样本存储量调整 D1 数据预算、测试 map seed 分割，以及正式环境步数。

## 官方案例/资料

- MetaDrive 安装：https://metadrive-simulator.readthedocs.io/en/latest/install.html
- MetaDrive 结构化观测：https://metadrive-simulator.readthedocs.io/en/latest/obs.html
- 环境与 train/test seeds：https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html
- 行为与专家策略：https://metadrive-simulator.readthedocs.io/en/latest/action.html
- 环境 API / 进程级并行：https://metadrive-simulator.readthedocs.io/en/latest/get_start.html
- Sensors 射线检测：https://metadrive-simulator.readthedocs.io/en/latest/sensors.html

**状态：只作资源与数据规划，不代表已完成 D0 试运行。**

## 服务器路径约定修正（2026-11-16）

用户依据既有 SMM 实验纠正：**数据区不在 `/home/ps/lxb` 下**。沿用以下历史绝对路径：

- SMM 项目工作区：`/home/ps/lxb/workspace/stochastic-microstate-memory`
- SMM self-hosted runner：`/home/ps/lxb/workspace/stochastic-microstate-memory/actions-runner`
- 共享数据区根目录：**`/mnt/datasets/lxb`**
- 既有 SMM 数据集：`/mnt/datasets/lxb/T-ImageNet200`
- 既有 SMM checkpoint 根目录：`/mnt/datasets/lxb/smm_checkpoints`

**建议 DLS 使用以下新路径（尚未在服务器验证或创建）：**

- 仓库：`/home/ps/lxb/workspace/drive-like-a-living-system`
- Runner：`/home/ps/lxb/workspace/drive-like-a-living-system/actions-runner`
- DLS 数据区：`/mnt/datasets/lxb/dls`
- 子目录：`metadrive_assets/`、`demonstrations/`、`scenarios/`、`checkpoints/`

以上把「过去实验已确认的路径」和「DLS 尚未创建的建议路径」明确区分。实际 MetaDrive Python 包依赖/资源管理路径需要在 D0 审计后设置，不假设环境变量或固定资源目录已经生效。


## 环境迁移决策（用户于 2026-11-16 授权）

- **环境放在系统 Conda 区，不放在数据盘**：
  `/home/ps/miniconda3/envs/dls-p0`。
- 用户明确授权卸载以下三个历史 Conda 环境以释放系统盘空间：
  `/home/ps/miniconda3/envs/lxb_ddpm-base`、
  `/home/ps/miniconda3/envs/lxb_edm1`、
  `/home/ps/miniconda3/envs/lxb_pmi-base`。
- 只删除上述三个精确路径；如果检测到任何仍在使用的进程，停止操作。
- 数据、仿真源代码与资源、专家示范、检查点、缓存等仍位于
  `/mnt/datasets/lxb/dls`。
- 对应自动部署工作流：[d0-env-migration.yml](../../.github/workflows/d0-env-migration.yml)。
- 本节记录授权和预期路径，**不表示卸载或新环境创建已经完成**；以 GitHub Actions 任务结果为准。
