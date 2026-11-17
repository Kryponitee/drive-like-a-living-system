# Drive Like a Living System (DLS)

**Current stage (2026-11-17): D0+D1 completed; D2a recurrent PPO math correctness passed; D2b small equal-step closed-loop PPO pilot running. No evidence yet for causal inter-module motor coordination.**

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

## D1 execution evidence

- [D1 shared expert dataset and A/B/C BC pilot design](docs/research/2026-11-17-D1-data-and-BC-pilot.md)
- [Completed 27-worker dataset collection](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44008816875)
- [A/B/C BC + closed-loop evaluation workflow — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44009291227)
- [Three-initialization repeat (42/43/44) — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44009829895)
- [Fresh held-out seeds 3100–3123 — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44010604782)
- [Fresh-scene test interpretation and detailed outcomes](docs/research/2026-11-17-D1f-fresh-scene-results.md): at density=0.10 C outperformed B for all three trained seeds, but aggregate advantage did not hold consistently over densities; no causal coordination conclusion.
- D2a: [recurrent PPO correctness audit — SUCCESS](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44011443700), old-policy replay probability ratio=1 with proper tanh density, recurrent state replays and finite gradients.
- D2b: [small 4,096-step-per-model PPO trial](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44011875754) launched, results must be checked before any learning claim.
- [D2 protocol and correctness record](docs/research/2026-11-17-D2a-D2b-recurrent-PPO.md).
- Future: replicated closed-loop optimization and frozen-weight direct-message causal perturbations (D3).

## Previous records (retained for provenance)

- [001 — Minimal definition and literature](docs/research/2026-11-16-motor-coordination-definition.md)
- [002 — Earlier sparse recurrent structure proposal](docs/research/2026-11-16-structure-proposal-v0.md)
- [003 — Perception integration P0 -> P1 -> P2](docs/research/2026-11-16-perception-integration-v0.md)
- [Earlier P0 sparse study draft (superseded as primary route)](docs/research/2026-11-16-P0-design-v0.md)
- [Modular review + inter-module coupling discussion (incorporated into P0 v1)](docs/research/2026-11-16-P0-modular-review.md)

The separate MALT / ResLadder research is not treated as evidence for DLS.
