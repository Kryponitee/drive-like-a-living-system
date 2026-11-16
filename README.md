# Drive Like a Living System (DLS)

**Current stage: P0 research protocol defined (2026-11-16); no implementation or experimental results.**

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
- **Status:** protocol only; D0 simulator/data audit is the next step, not yet executed.

## Previous records (retained for provenance)

- [001 — Minimal definition and literature](docs/research/2026-11-16-motor-coordination-definition.md)
- [002 — Earlier sparse recurrent structure proposal](docs/research/2026-11-16-structure-proposal-v0.md)
- [003 — Perception integration P0 -> P1 -> P2](docs/research/2026-11-16-perception-integration-v0.md)
- [Earlier P0 sparse study draft (superseded as primary route)](docs/research/2026-11-16-P0-design-v0.md)
- [Modular review + inter-module coupling discussion (incorporated into P0 v1)](docs/research/2026-11-16-P0-modular-review.md)

The separate MALT / ResLadder research is not treated as evidence for DLS.
