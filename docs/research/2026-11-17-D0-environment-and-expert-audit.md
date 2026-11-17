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


## GPU 运行时审计（已完成）

GitHub Actions #44004713319：**SUCCESS**。在系统盘 Conda 环境 `/home/ps/miniconda3/envs/dls-p0` 中完成：

- PyTorch `2.5.1+cu124`，CUDA Runtime 12.4；
- TensorBoard `2.21.0`；
- 检出 8 张 NVIDIA GeForce RTX 4090；
- `torch.cuda.is_available() == True`，单卡 CUDA 上随机矩阵乘法及有限值断言通过；
- `pip check` 无 broken requirements。

报告位置：`/mnt/datasets/lxb/dls/reports/d0/gpu_runtime.json`；
安装依赖冻结：`/mnt/datasets/lxb/dls/reports/d0/pip_freeze.txt`。

这仅验证 **GPU 计算与 Python 环境**，不等于 A/B/C 控制模型、训练或模块化协调实验已完成。

## D0c 后续数据覆盖检查（已启动，待结果）

对同 12 个 pilot seeds，保持专家/地图/800 步上限不变，仅将交通密度在 0.00 / 0.10 / 0.25 切换。分别报告成功、碰撞、超时、转向/制动动作比例。**动作比例仅是覆盖代理，不能证明发生了跟车、换道或扰动恢复。**

- Workflow: https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44005436462
- 结果文件（待验证）：`/mnt/datasets/lxb/dls/reports/d0/expert_traffic_coverage.json`
- 成功轨迹候选：`/mnt/datasets/lxb/dls/demonstrations/d0c_success_only/`


## D0c 专家交通密度控制实验（已完成）

原始日志：https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44005436462

- 固定：MetaDrive 0.4.3，map=3，deterministic PPO expert，seed 0–11，最大 800 步。
- 仅操纵交通密度：0.00、0.10、0.25。
- **独立变量与结果**：

| 交通密度 | 到达 | 碰撞 | 超时 | 转向激活步占比 (\|steer\|>0.1) | 刹车步占比 (pedal<-0.05) | 成功 episode 样本步数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 12/12 | 0/12 | 0/12 | 0.117 | 0.132 | 4,230 |
| 0.10 | 9/12 | 3/12 | 0/12 | 0.111 | 0.144 | 4,129 |
| 0.25 | 6/12 | 3/12 | 3/12 | 0.131 | 0.151 | 3,321 |

**结论（仅限测试配置）**：专家在无交通驾驶可达到 12/12，但交通越密集，到达率越低；在当前 12 seed 下 density=0.25 额外出现 3 个超时。当前专家适合作为基础道路跟踪行为的示范来源，不能直接视为高质量的全面交通交互专家。

**重要限制**：`pedal<-0.05` 的动作比例只是制动行为的代理：即使 density=0 时也有 13.2% 的步数触发，不能据此认定专家在执行跟车制动；是否有实际前车交互仍需特权物理状态的只读诊断。36 场景样本量不足以精确估计置信区间和跨地图泛化。

- 报告：`/mnt/datasets/lxb/dls/reports/d0/expert_traffic_coverage.json`
- **只保存成功轨迹候选**：`/mnt/datasets/lxb/dls/demonstrations/d0c_success_only/`，共 11,680 转移样本。
- D0d 交互语义诊断：https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44007689347 （当下执行中；结果尚未审查）。

**D1 的门槛**：仅获得成功 episode 不足以证明覆盖目标任务。需确认“成功轨迹中真实存在何种前车/近车交互”和独立训练测试划分，然后才生成正式大样本 BC 数据。


## D0d 真实前车暴露诊断（已完成）

工作流：https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44007972328  
输出：`/mnt/datasets/lxb/dls/reports/d0/expert_interaction_exposure.json`

固定：同一批 12 个场景种子、`map=3`、800 步上限、确定性专家；仅增加对仿真内部物理状态的**只读统计**，不引入模型输入、不训练。

- 同车道前车：`lane_index` 相同，前车中心沿自车所在车道的纵坐标差在 (0, 50m]；
- 接近前车：前车中心在 40m 内且自车速度较前车高出 5km/h 以上；
- 这些只是几何/速度代理指标，不表示发生了经过验证的前车减速、避让或反馈补偿。

| 指标 | traffic_density=0.10 | traffic_density=0.25 |
|---|---:|---:|
| 到达 | 9/12 | 6/12 |
| 任何时刻遇到 50m 内同车道前车的场景 | 7/12 | 9/12 |
| **成功场景中曾遇同车道前车** | **7/9** | **6/6** |
| 处于接近前车条件的步数比例 | 3.75% | 7.66% |
| 前车在 50m 内的制动动作比例 | 10.36% | 12.15% |
| 无同车道前车时的制动比例 | 16.18% | 18.46% |

**科学判断：** 目前的成功示范不仅是空车道行驶；在此几何定义下确有同车道前车暴露（0.10 时 7/9、0.25 时 6/6 的成功 episode）。但是“前车出现”和“专家制动”并没有表现出直接的简单正关联；实际的前车突然减速响应、变道、交汇和因果恢复仍**未被证实**。不能因为有近车/接近步数就宣称 P0 所需的复杂交互技能覆盖已满足。

**D1 准入建议：** 可以开始**受限的 BC pilot**，用于学习道路跟踪+部分交通邻车状态下的基本行为；必须保留独立场景种子测试，正式数据应扩大种子数、成功轨迹分层筛选，困难交通场景需留给 D2 闭环交互，不要把 D1 数据覆盖与 D3 协调机制证据混为一谈。
