# DLS P0 — Official RecurrentPPO 300k continuation results

Date: 2026-11-17. Authoritative [D2j GitHub Actions #44025266123](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44025266123), **SUCCESS**.

## Protocol

- Reused the successful D2i official SB3-Contrib 2.6.0 `MlpLstmPolicy` checkpoint after 16,384 env frames and its original `VecNormalize` statistics. **No retraining from scratch**, no updates to DLS A/B/C weights and no new dependency installs.
- Official SB3-Contrib RecurrentPPO on 4 independent headless MetaDrive 0.4.3 CPU environments (map=3, traffic density=0.1), each worker on its own training seed range: 6000..6079, 6080..6159, 6160..6239, 6240..6319.
- Added 283,648 frames to reach **300,032 aggregate environment frames**. Training `reset_num_timesteps=False`; 4 workers, rollout128 steps per worker, minibatch128, 4 PPO epochs, Gamma0.99, GAE0.95, exploration log_std initially -3.
- Evaluated both the frozen original 16k and extended 300k weights on the **same previously unused 24 scenarios** seeds5500..5523. Each weight evaluated using its own saved/frozen observation-normalization statistics and raw environment reward. No model selection on these scenes.

## Verified results

| Metric | Source checkpoint | Extended checkpoint |
|---|---:|---:|
| Aggregate training frames | 16,384 | 300,032 |
| Paired new-scene successes | 0/24 | **2/24** |
| Additional successes (new only) | — | **2** |
| Post-training mean raw environment reward | not in submitted stdout (available in JSON) | **165.68** |
| Training workers | 4 | 4 |

- Final throughput logged in training ~**185 environment frames/sec**; earlier windows varied ~184–205.
- PPO logged values explained variance generally high late in training (e.g. near final ~0.803), computed under **SB3's normalized training targets**, not comparable to D2g's unnormalized actual discounted-return EV.
- 300k checkpoint and running statistics were saved successfully and verified at the end of the job.
- This performance is **not good enough** to call a learned reliable driving policy. It only proves an official mature training implementation is able to run over a substantial budget, plus a small initial improvement on matched new scenes.

## Interpretive boundary and prioritized next step

The D2i/D2j LSTM control differs from our custom A/B/C **in architecture, BC initialization, observation normalization, training budget, and data occupancy**. Therefore it is a system-sanity reference, not a fair architecture control and does not prove or disprove direct inter-module coordination.

The server has many idle CPU threads; the 4-worker count was inherited from the initial smoke instead of being tuned. Run the pre-registered [D2k 4/8/16/32 scaling benchmark](2026-11-17-D2k-worker-scaling-plan.md) using **this same frozen 300k checkpoint** to determine throughput before choosing large PPO budgets. Benchmarked environment sampling throughput is not equivalent to PPO optimizer-inclusive throughput.

The next scientific step is still capacity-/initialization-/budget-matched A/B/C under stable closed-loop learning, followed by module-message interventions; do not spend additional hundreds of thousands of steps on this weak randomly initialized LSTM just to hope for better scores.

## Files on the data disk

- `/mnt/datasets/lxb/dls/reports/d2j/official-ppo-300k.json`
- `/mnt/datasets/lxb/dls/checkpoints/d2j/official-recurrentppo-300k.zip`
- `/mnt/datasets/lxb/dls/checkpoints/d2j/official-recurrentppo-vecnormalize-300k.pkl`
- `/mnt/datasets/lxb/dls/tensorboard/d2j/`

The core `dls-p0` Conda environment and all original A/B/C checkpoints remain unchanged.
