# DLS P0 — D0 仿真与专家质量审计结果

**日期：2026-11-17（北京时间）**  
**状态：已在 GitHub Actions self-hosted Runner 实际运行。**  
**不是模型训练结果；尚未启动 D1。**

## 运行环境（实测）

- Runner: `DLS`，服务器 `pow`，无图形界面执行。
- Conda：`/home/ps/miniconda3/envs/dls-p0`，Python 3.10.14。
- MetaDrive：PyPI `metadrive-simulator==0.4.3`。
- 官方配套资源版本：0.4.3，下载源 `gh-proxy.com`（官方站点下载速度过慢后自动切换）；资源目录 `/mnt/datasets/lxb/dls/metadrive_assets`。
- ZIP 文件：134,074,203 字节，ZIP 结构校验通过，下载文件 SHA256：`4f0da9f5143a1258131c5b55f77bdf170c0f9bce8a9f18dd41b3678df779eac9`。这是**本次下载计算得到的哈希**，未与官方发布的独立签名/校验和比对。
- Headless smoke：seed 0/1/0，分别 40 步，状态为 259 维 `float32`、动作空间 `Box(-1,1,(2,),float32)`，seed 0 重置观测复现检查通过。
- 环境部署任务： https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44002549428

## D0 专家策略审计

统一设置：MetaDrive 0.4.3、`traffic_density=0.1`、`map=3`、deterministic PPO NumPy expert、场景 seeds 0–11。专家执行数据只写到数据盘，未经专家质量筛选不可直接用于正式训练。

| 指标 | 首轮（最大 400 步） | D0b（最大 800 步） |
| --- | ---: | ---: |
| 场景数 | 12 | 12 |
| 到达目的地 | 4/12 (33.3%) | 9/12 (75.0%) |
| 碰撞车辆 | 3/12 (25.0%) | 3/12 (25.0%) |
| 越界 | 0/12 | 0/12 |
| 未到终点且无碰撞（上限截断） | 5/12 | 0/12 |
| 采集连续转换 | 3,916 | 4,609 |
| 259维观测 | PASS | PASS |
| 数据包大小 | 747,328 bytes | 881,262 bytes |

- **结论：原 400 步截断人为低估了专家完整驾驶成功率**。延长上限后，5 个原本截断的场景均成功到达，3 个原本碰撞的场景仍然碰撞（没有观察到事故改善）。
- 上述 12 个场景是初步诊断，不能据此推断全部地图/所有交通条件的成功率、教师泛化能力或真实驾驶安全性。
- 现阶段 **只保留成功完成的 episode 作为第一批 BC 候选示范**；碰撞轨迹另存/标识供失败分析，不应不加筛选与成功轨迹混合。
- 对道路跟踪、跟车、交互场景分层检查覆盖率仍未完成；不能宣称“专家已经适合全部 P0 场景”。
- 400 步审计：https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44004336636
- 800 步审计：https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44004597623

## 文件位置

- 基础验证：`/mnt/datasets/lxb/dls/reports/d0/smoke.json`
- 原始审计：`/mnt/datasets/lxb/dls/reports/d0/expert_audit.json`
- 400 步样本：`/mnt/datasets/lxb/dls/demonstrations/d0-expert-probe.npz`（初始审计格式）
- 800 步报告：`/mnt/datasets/lxb/dls/reports/d0/expert_audit_h800.json`
- 800 步样本：`/mnt/datasets/lxb/dls/demonstrations/d0-expert-probe-h800.npz`

**下一步**：评估 9 个成功 episode 是否覆盖 P0 要研究的运动情境，并基于成功轨迹、独立场景划分生成正式 D1 教师数据；在此之前不从这些小样本得出学习或协调结论。GPU PyTorch 运行时另行验证。
