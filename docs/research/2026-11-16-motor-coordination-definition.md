# DLS research record 001 — Minimum definition of learned motor coordination

- Date: 2026-11-16
- Stage: literature review + modeling; **not** an implemented method or experimental result
- Trigger: user requested that the coordination structure be learned from data rather than defined as a fixed list of motor units
- Prior inspiration: SparseDriveV2 motivates questioning trajectory as the principal learning object; it does **not** demonstrate emergent motor coordination.

## 1. The claim we want to test

Can a neural driving system learn a **reusable, feedback-responsive organization of motor control** from driving experience rather than preassigning the semantics of its units?

Do **not** require a unique biological answer, a fixed number of discovered units, sparsity, or correspondence to muscles.

## 2. Minimal external interface

Let $s_t$ be the physical state, $o_t$ the observation, $g_t$ the task/navigation condition, $z_t$ a high-level coordination signal, $h_t$ the motor system's latent internal state, and $u_t$ the vehicle command:

$
z_t=N_\theta(o_t,g_t,f_t),\qquad
(h_{t+1},u_t)=C_\phi(h_t,s_t,z_t),\qquad
s_{t+1}=F(s_t,u_t,\xi_t).
$

$f_t$ summarizes execution feedback. $F$ describes vehicle/environment dynamics and perturbation $\xi_t$.

This interface allows **hierarchical feedback** (with timescales to be defined), but neither fixes a GRU/DiT nor implies distinct motor units have been learned.

The latent state $h_t$ can implement learned interactions or modular dynamics, but no particular factorization of (h_t) is asserted at this stage.

## 3. Operational definition: what counts as meaningful coordination?

**Proposed research definition (not a textbook equivalence):** A learned motor coordination *structure* is a reproducible organization of internal dynamic components whose interactions support task-relevant stabilization under feedback and can be reused under changed contexts.

Three gates:

1. **Learned organization**: organization is inferred/optimized from experience, rather than assigned skill labels or fixed functional roles. Fixed model capacity or an inductive bias is permissible; its role must be documented.
2. **Task-conditioned reusable behavior**: some learned organization retains a useful control role when context/initial speed/geometry changes, without re-learning the full controller. Evaluate with matched baselines, not by inspecting a single latent coordinate.
3. **Functional feedback compensation**: in controlled perturbations, internal components jointly adjust and stabilize designated task variables more reliably than relevant ablations. Attribute effects by *interventions*, not correlations.

Merely obtaining low-dimensional states, attention weights, different-looking modules, or high driving scores is **insufficient**.

### Task variables, not unique trajectories

Let $q(s)$ represent task-relevant quantities (e.g. clearance to collision, lateral position within drivable area, speed relative to a context-dependent requirement). Different motions can satisfy the same goal.

A candidate controlled quantity is

$
J_q(\pi;\mathcal D)=\mathbb E_{\mathcal D,\,\pi}\left[\sum_t \ell_q(q(s_t),g_t)\right],
$

measured across context and disturbance distributions. This is an evaluation template, **not** a prescribed training reward or a formal safety guarantee.

No uniquely identifiable internal decomposition is required. An invertible reparameterization may yield the same outputs; what matters is the reproducible *functional role* and effect of perturbing an organization, acknowledging such roles may be coordinate-dependent.

## 4. Literature findings and limits

| Work | Supported finding | What it does NOT establish |
|---|---|---|
| Scholz & Schöner (1999), Uncontrolled Manifold | Characterizes stabilization of task-relevant variables in motor coordination | Does not provide a directly trainable driving network |
| Todorov & Jordan (2002), Optimal Feedback Control | Feedback can prioritize task-relevant errors and tolerate variability in redundant dimensions; synergies can emerge in the model | Does not imply a learned neural driving policy automatically develops separate units |
| Latash, Scholz & Schöner (2007), Motor Synergy | Functional synergy relates one-to-many motor mappings, task stability, and perturbation flexibility | Does not prescribe a number of car control units |
| Haruno, Wolpert & Kawato (2001), MOSAIC | Paired predictive/controlling models and contextual/feedback selection can be learned | Module architecture/candidate set is designed; not full unconstrained structure discovery |
| Goyal et al. (2021), RIM | Sparse recurrent module activation and communication can induce specialization | Number and layout of candidate mechanisms are imposed by the architecture |
| Kipf et al. (2018), NRI | Latent interaction relationships between given entities can be inferred from dynamics | Assumes entities (nodes); not free-form creation of motor organs |
| Locatello et al. (2019), Disentanglement | Arbitrary factors cannot generally be uniquely recovered without inductive bias | Does not rule out task-defined, non-unique useful control structures |
| Lynch et al. (2020), Play-LMP | Self-supervised latent organization can support reusable reactive skills from varied experience | Does not by itself prove biologically analogous motor coordination |

### Verified reading links

1. Scholz & Schöner (1999): https://pubmed.ncbi.nlm.nih.gov/10382616/
2. Todorov & Jordan (2002): https://www.nature.com/articles/nn963
3. Latash, Scholz & Schöner (2007): https://pubmed.ncbi.nlm.nih.gov/17715460/
4. Haruno, Wolpert & Kawato (2001): https://pubmed.ncbi.nlm.nih.gov/11570996/
5. Goyal et al. (2021): https://mlanthology.org/iclr/2021/goyal2021iclr-recurrent/
6. Kipf et al. (2018): https://proceedings.mlr.press/v80/kipf18a.html
7. Locatello et al. (2019): https://proceedings.mlr.press/v97/locatello19a
8. Lynch et al. (2020): https://proceedings.mlr.press/v100/lynch20a.html

Sources above were reviewed at the abstract/overview level; this is **not** a comprehensive novelty search, theorem reconstruction, or independent reproduction.

## 5. Discriminating comparisons (design only, no experiments yet)

- **Ordinary recurrent end-to-end policy** with matched parameters/data/feedback: high score alone is not evidence of learned coordination.
- **Candidate modular policy** with learned interactions but *no assigned semantics*: asks whether stable functional organization appears.
- **Remove or disrupt internal interactions after learning**: measures whether interactions contribute causally to task-relevant stability; control for matched perturbation magnitude and lost capacity.
- **Transfer the same coordination mechanism to held-out contexts**: separates reusable structure from memorized motion.
- **Disable/alter feedback at matched information and compute budgets**: distinguishes reactive control from merely replaying motion.

Potential quantitative evidence: relative task loss under disturbances, collision/road-boundary rates, recovery time, transfer gap, and interaction-ablation effect. Pre-register task variables and comparison budgets once a benchmark is selected.

### Failure/decision rules

- No stable *intervention-sensitive* organization: do not claim synergy; revisit identifiability/data/bias.
- Organization found but no measurable transfer or feedback benefit: report limited functional value, no overclaim.
- Gains disappear under capacity/compute matching: reject architecture-specific benefit.
- An ordinary recurrent policy achieves the same robustness: do not label generic feedback robustness as motor synergy.

## 6. Next modeling gate (NOT approved yet)

Choose the **smallest learnable structure hypothesis class**, rather than hand designing eight GRU "muscles". Options to assess rigorously:
- continuous state with emergent organization, no explicit units (strong functional test; weak interpretability);
- soft modular dynamics/communication with learned gating (some architectural bias);
- learned latent interaction graph (requires definition of entities/nodes).

Decision criterion: which choice admits meaningful intervention + reuse tests while minimizing hand-specified semantics and excessive complexity?

No software stack, module count, training algorithm, test benchmark, or simulator is selected by this record.
