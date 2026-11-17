# Drive Like a Living System (DLS)

**Current stage (2026-11-17): D0 completed; D1 expert collection verified (91,717 train + 9,822 validation transitions); three-model BC pilot running. Coordination mechanism evidence not yet available.**

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
- D0 traffic/interaction audits completed; proceed with the bounded D1 pilot (do not assume module coordination is established).

## D1 execution evidence

- [D1 shared expert dataset and A/B/C BC pilot design](docs/research/2026-11-17-D1-data-and-BC-pilot.md)
- [Completed 27-worker dataset collection](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44008816875)
- [A/B/C BC + closed-loop evaluation workflow](https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44009291227) (running as of latest update; no results assumed)

## Previous records (retained for provenance)

- [001 — Minimal definition and literature](docs/research/2026-11-16-motor-coordination-definition.md)
- [002 — Earlier sparse recurrent structure proposal](docs/research/2026-11-16-structure-proposal-v0.md)
- [003 — Perception integration P0 -> P1 -> P2](docs/research/2026-11-16-perception-integration-v0.md)
- [Earlier P0 sparse study draft (superseded as primary route)](docs/research/2026-11-16-P0-design-v0.md)
- [Modular review + inter-module coupling discussion (incorporated into P0 v1)](docs/research/2026-11-16-P0-modular-review.md)

The separate MALT / ResLadder research is not treated as evidence for DLS.
