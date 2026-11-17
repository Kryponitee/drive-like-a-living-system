# Drive Like a Living System (DLS)

**Current stage (2026-11-17): D0/D1 and official PPO infrastructure are complete. D2n-A BC-initialized PPO did not consistently improve driving. D3a/D3a2/D3c froze BC C and tested learned cross-module routing/content: no robust dependence over three independently trained seeds (only seed44 showed candidate loss on self-only substitution). D3d/D3e directly tested hidden-state retention: even when the entire fast-GRU/slow-z memory is discarded every tick, C seed42/43 maintained similar or better driving, while seed44 was harmed. Conclusion: the current 259D structured-observation BC task has not provided replicated evidence that C needs direct communication or persistent memory. Next high-value controlled experiment: a separately trained capacity-/data-matched feedforward BC baseline (D4a), before redesigning partial-observability/task temporal demands.**

## Research question

Can learned, reusable motor coordination emerge from driving experience and improve hierarchical closed-loop feedback control?

We **do not** prescribe driving-skill labels to internal units or assume that a modular architecture guarantees motor synergy.

## Active protocol

**[P0 v1.0 — Learnable modular motor coordination with feedback and communication](docs/research/2026-11-16-P0-protocol-v1.md)**

- **Environment:** MetaDrive structured observations (no camera/visual E2E in P0).
- **High level N:** scene/navigation/feedback -> coordination conditioning.
- **Low level C:** learned recurrent modules, soft routing, common action head, physical-state feedback.
- **Main comparison:** modular controller with *only shared physical feedback* versus the same controller *with learnable direct inter-module communication*. Include a matched monolithic recurrent-policy baseline.
- **Evidence requirements:** closed-loop competence, repeatable role/reuse, and causal feedback compensation under controlled perturbations.
- **Status:** MetaDrive 0.4.3 environment and headless smoke verified. Deterministic expert audit: 9/12 success with 800-step horizon versus 4/12 at 400-step cap; 3/12 collision in either setting. See the report below. This is an infrastructure/data audit, **not a learned coordination result**.

## D0 execution evidence

- [2026-11-17 — Environment and expert audit (real server logs and outcomes)](docs/research/2026-11-17-D0-environment-and-expert-audit.md)
- Dataset and asset root: `/mnt/datasets/lxb/dls/` on a separate volume.
- Python Conda environment: `/home/ps/miniconda3/envs/dls-p0/` on system disk.
- D0 traffic/interaction audits completed; training and evaluation outputs stored exclusively on the data disk.

## D1 and D2 execution evidence

- [D1 shared expert dataset and A/B/C BC pilot design](docs/research/2026-11-17-D1-data-and-BC-pilot.md)
- [Completed 27-worker dataset collection](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44008816875)
- [A/B/C BC + closed-loop evaluation workflow — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44009291227)
- [Three-initialization repeat (42/43/44) — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44009829895)
- [Fresh held-out seeds 3100–3123 — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44010604782)
- [Fresh-scene test interpretation and detailed outcomes](docs/research/2026-11-17-D1f-fresh-scene-results.md): at density=0.10 C outperformed B for all three trained seeds, but aggregate advantage did not hold consistently over densities; no causal coordination conclusion.
- D2a: [recurrent PPO correctness audit — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44011443700), old-policy replay probability ratio=1 with proper tanh density, recurrent state replays and finite gradients.
- D2b: [equal-step 4,096-transition PPO pilot — completed](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44011875754): train-time success episodes A/B/C = 0/0/3; final closed-loop performance broadly degraded.
- D2c: [frozen BC noise intervention — completed](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44012974852): at density 0.10 success A/B/C collapsed from 16/18/18 of 24 when deterministic to 1/1/1 at pre-tanh exploration sigma=0.27.
- D2d: [one-factor sigma=0.05 PPO correction — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44013255699); training successes A/B/C changed from 0/0/3 to 3/4/3, but final driving did not improve beyond BC.
- D2e: [read-only reward/imitation audit — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44014040894); at traffic density 0.10 PPO reduced both episodic reward and success while increasing BC validation imitation error.
- D2f: [BC functional anchor PPO run](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44015006851): training, paired evaluations and validation passed; only last aggregate step failed due incorrect top-level seed metadata.
- [D2f audit-only corrected aggregate — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44016058139): validates 30 completed original per-episode scene IDs, preserving untouched raw results. No reruns.
- [D2f result and interpretation](docs/research/2026-11-17-D2f-BC-functional-anchor-results.md): validation Huber improved A/B/C, but new-scene closed-loop benefit arose only in B, not consistently across models.
- D2g: [frozen critic calibration audit — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44016459801); six frozen D2d/D2f critics explain only ~0.8–1.9% of observed discounted return variation, with value standard deviation ~0.7 vs realized returns ~24–26.
- [D2g results and measured limits](docs/research/2026-11-17-D2g-critic-calibration-results.md).
- D2h: [value-only critic supervision — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44023029712): independent validation EV A/B/C = +21.4% / +16.2% / −15.5%; constant-baseline RMSE 23.77/24.75/25.18 vs new critic 22.79/23.76/30.07. Actor frozen, no PPO updates.
- [D2 protocol and correctness record](docs/research/2026-11-17-D2a-D2b-recurrent-PPO.md).
- Future: replicated closed-loop optimization and frozen-weight direct-message causal perturbations (D3).

- [Official MetaDrive/SB3-Contrib/TorchRL comparison and D2h pre-registration](docs/research/2026-11-17-official-RL-stack-audit.md): standard RL infrastructure should be reused; SB3's LSTM baseline is not a drop-in replacement for custom A/B/C recurrent modules.
- [D2i matched Critic history ON/OFF plan](docs/research/2026-11-17-D2i-critic-history-control-plan.md) and [confirmed null final result](docs/research/2026-11-17-D2i-history-critic-results.md): no training data recaptured; 3 model occupancies × exact same GRU parameters and fixed 50 epochs; 0/3 met 5% gain criterion. [Actions SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44024937004).
- [Official RecurrentPPO 16k smoke — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44024379218): four parallel MetaDrive workers + standardized RecurrentPPO pipeline; 0/12 arrivals after 16k frames, no convergence claim. [Technical result and dependency isolation](docs/research/2026-11-17-official-recurrentppo-smoke-results.md).
- [D2j official RecurrentPPO 300k-frame continuation — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44025266123): reuses checkpoint and VecNormalize; 300,032 total frames, 4 CPU workers, 2/24 new-scene successes vs 0/24 at 16k. [D2j result and limitations](docs/research/2026-11-17-D2j-official-300k-results.md).
- [D2k initial worker-scaling — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44028935956): benchmark 4/8/16/32 concurrent CPU MetaDrive simulators; [performance protocol](docs/research/2026-11-17-D2k-worker-scaling-plan.md).
- [D2k extended 32/48/64 workers — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44029425146): matched 24,576 steps each; throughput 551.3/759.1/925.4 steps/s. **64 workers selected** for subsequent optimizer-inclusive tests; [full performance results and limitations](docs/research/2026-11-17-D2k-extended-worker-scaling-results.md).


- [D2l full-PPO 32 vs 64 worker optimizer-inclusive test — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44030181386): equal 32,768 frames and 1,024 actual optimizer updates; full training FPS **224.0 / 258.2**. [Full report](docs/research/2026-11-17-D2l-full-PPO-worker-results.md).
- [D2m async state/slow-clock validation — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44030866838) and [official custom recurrent policy PPO bridge — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44031524331): A/B/C deterministic BC parity, replay error 2.62e-5 with official sequence buffer, real MetaDrive PPO updates. [Technical result](docs/research/2026-11-17-D2m-official-native-policy-results.md).
- [D2n-A first official A/B/C BC-initialized PPO training — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44032089904): each 65,536 frames / 64 workers, 24 new paired scene seeds × 3 densities. Primary density0.10 success A 11→12, B 13→13, C 13→9; C average raw reward 268.90→237.21. [Full experimental result](docs/research/2026-11-17-D2n-A-results-and-D3-transition.md).
- [D2n-A read-only A/B runtime audit — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44034618614): A/B environment collection 85.2/94.4s but PPO + other non-collection 209.1/611.6s. Computational overhead is principally non-collection; parameter counts are matched but cost is not.
- [D3a frozen BC B/C message-recipient routing — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44036737339): C original→rotated success 9→11 at density0.10 and 4→4 at density0.25, no consistent loss. [Outcome and limits](docs/research/2026-11-17-D3a-message-recipient-results.md).
- [D3a2 direct C cross-message content replacement — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44037421039): per-recipient matched-norm own-message substitute; 48 new episodes only, reusing D3a C original/rotation. C medium-density successes 9→12, rewards 284.63→296.03, but high-density successes 4→3. [Full interpretation](docs/research/2026-11-17-D3a2-cross-message-content-results.md).
- [D3c 3-seed frozen C BC cross-message replication — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44038017935): existing C training seeds42/43/44 × 24 new paired scenes × original/own-only, reward changes +21.36/−5.92/−26.23, only1/3 passed preregistered dependence criterion. [Measured result](docs/research/2026-11-17-D3c-three-seed-cross-message-results.md).
- [D3d frozen C GRU fast-history reset — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44038662180): seed42/43/44 original→h-zero successes 14→15 / 12→15 / 15→14; no stable fast-memory dependency. [Full report](docs/research/2026-11-17-D3d-fast-memory-reset-results.md).
- [D3e strictly current-observation-only C policy — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44039281732): reset both GRU h and slow z EVERY tick using unchanged pretrained weights. BC seed42/43/44 original→stateless successes 14→15 / 12→12 / 15→13. [Full result, limits and next high-value experiment](docs/research/2026-11-17-D3e-stateless-reactive-results.md).
- **D4a capacity-matched feedforward BC control — workflow submitted, result pending**: independently train stateless F (114477 params) vs frozen C (114390), reuse D1 BC data and three training seeds; 24 new scenes × 3 densities × F/C × 3 seeds, 54 CPU evaluators. [Pre-registered scientific design](docs/research/2026-11-17-D4a-capacity-matched-feedforward-plan.md); [Actions workflow](https://github.com/Kryponitee/drive-like-a-living-system/actions/workflows/d4a-feedforward-control.yml). Do not claim results before run verification.

## Previous records (retained for provenance)

- [001 — Minimal definition and literature](docs/research/2026-11-16-motor-coordination-definition.md)
- [002 — Earlier sparse recurrent structure proposal](docs/research/2026-11-16-structure-proposal-v0.md)
- [003 — Perception integration P0 -> P1 -> P2](docs/research/2026-11-16-perception-integration-v0.md)
- [Earlier P0 sparse study draft (superseded as primary route)](docs/research/2026-11-16-P0-design-v0.md)
- [Modular review + inter-module coupling discussion (incorporated into P0 v1)](docs/research/2026-11-16-P0-modular-review.md)

The separate MALT / ResLadder research is not treated as evidence for DLS.
