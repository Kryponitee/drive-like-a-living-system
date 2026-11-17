# DLS D2k — MetaDrive CPU workers scaling benchmark (pre-registered)

Date: 2026-11-17. Run only after D2j official 300k baseline is verified.

## Question
On the actual DLS host with many available logical CPU threads, which of 4, 8, 16 and 32 independent MetaDrive processes gives the highest usable **environment inference/sampling throughput**, using the same trained control policy and total physical simulator step budget?

D2i-D2j carried over a smoke-test configuration with **4 workers**. This was safe but may waste CPU capacity. More workers are **not guaranteed** to scale linearly due to Panda3D startup, memory contention, PyTorch thread pools and inter-process communication.

## Controlled benchmark
- Same **frozen** [official D2j SB3-Contrib RecurrentPPO 300k](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44025266123) checkpoint and saved `VecNormalize` observation statistics in all 4 configurations.
- Headless MetaDrive0.4.3, map3, traffic_density0.1, max horizon800, 259D structured observations, original 2D continuous action.
- 4/8/16/32 independent `SubprocVecEnv` workers created with `spawn`, each in its own MetaDrive seed interval (>=64 scenarios per worker). No shared Panda3D engine.
- For each configuration, initialize workers, reset, warm up with 2 vector steps, then measure exactly **8,192 environment transitions** (fixed aggregate sample count, not 8,192 per worker); same model loaded from disk, deterministic recurrent policy inference.
- Record env initialization time separately, measured wall seconds, aggregate steps/sec, mean per-worker steps, completed episodes, per-worker process resident memory (RSS via psutil), host available RAM and failure codes. Each configuration runs **sequentially**, so no cross-trial resource competition.
- Freeze saved model and normalization; no PPO updates, no expert demonstration collection, no updates to A/B/C or D2j weights. Report is a throughput decision aid, not an architecture or scientific experiment.
- Selection gate: choose the highest stable throughput with adequate memory. Where speed improvement is below 10%, prefer the smaller worker count. Do not assume a 4x worker increase yields a 4x training speed increase.
- Important limit: this is **sampling with policy inference**, not optimizer-inclusive PPO throughput; use a subsequent short optimizer-inclusive benchmark on the selected configuration before allocating a large formal training budget.

## Data and execution
GitHub workflow: `.github/workflows/d2k-worker-scaling.yml`; data root `/mnt/datasets/lxb/dls`; isolated root-disk `dls-sb3-compat` environment reused unchanged. The resulting JSON goes under `reports/d2k`.

Scientific gate after throughput tuning: **official random-initialized RecurrentPPO only reached 2/24 successes after 300k frames**. This is a weak reference policy, not validation of A/B/C coordination. Do not simply increase the training budget without paired objective/teacher/architecture controls; focus subsequent controlled learning on BC-initialized A/B/C while reusing mature RL infrastructure.
