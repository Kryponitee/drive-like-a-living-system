# DLS D3a — Frozen-BC module-message recipient identity intervention (preregistered)

Date: 2026-11-17. Decision follows completed D2n-A official PPO and paired new-scene testing. **This document precedes D3a outcomes.**

## What we have established

D2n-A trained A/B/C through the official recurrent PPO 65,536 frames/model, 64 workers, all jobs passed. At traffic density 0.10 on seeds 30000..30023, BC→PPO success was A 11→12, B 13→13, C 13→9. Average raw reward B 261.64→259.51 and C 268.90→237.21. Thus PPO did not establish a consistent improvement; expanding to three PPO training seeds now would not directly advance the central research question. Reuse existing **D1 seed42 frozen BC** B/C as viable driving controls; D1 three-seed BC work already exists and later replication must incorporate it.

## One falsifiable point

**Does the identity of the receiving module matter causally for the policy's behavior, beyond the existence/magnitude of activity or communication weights?**

The normal C module messages are `aux[:,i,:]` computed from other modules' hidden states and learned attention. The only intervention at every control step is to **cyclically rotate message recipients**:

`aux_intervened[:, i, :] = aux_original[:, (i-1) mod 4, :]`

This preserves the multiset of produced message vectors, every-step total L2 energy, model weights, current sensor observations, routing head, and action decoder. It changes which module receives each message, and necessarily alters the subsequent closed-loop state trajectory. No learned parameter or training data modification.

To avoid conflating routing selectivity with nonspecific perturbation sensitivity, apply the **same recipient rotation** to B's local-only aux branch as a structural control. IMPORTANT: B's rotated aux deliberately violates B's normal self-only wiring and can introduce artificial cross-module inputs. Therefore B is a **sensitivity control**, *not* a clean communication-negative architecture after intervention. Difference-in-differences cannot be treated as proof of communication advantage.

Conditions: B-baseline, B-rotate, C-baseline, C-rotate. Original D1 BC seed42 checkpoint for every condition. No PPO, no new expert data, no resetting memory during episodes beyond actual MetaDrive done.

## Verification and fixed evaluation

1. **Implementation parity must be exact:** the copied no-rotation step formula must match unmodified `ModularController.step` using the same checkpoint for 32 synthetic steps per model, including slow-loop updates and hidden state; threshold 2e-6. Reject run if parity fails. The intervention only adds a `torch.roll(..., shifts=1, dims=1)` on aux.
2. On randomly sampled hidden states, assert shape `[N,4,32]`, exact equality of flattened per-time norm sum and message multiset under rotation; ensure the changed messages are not trivially identical for all targets.
3. Independent new MetaDrive scenes `31000..31023` (24) at densities `0.10` (primary) and `0.25` (stress), map3, horizon800; paired scenarios and deterministic actions. Four conditions ×2 densities ×24 episodes=**192 closed-loop episodes**, run as eight independent CPU processes. Do not use prior D2n-A evaluation seeds for the primary new result.
4. Output success, collision, road departure, raw reward, route completion, per-seed paired reward/success differences. Include mean absolute original-message energy and per-step norm-invariance audit. Source checkpoint SHA-256 must remain unchanged.
5. Main diagnostic: compare C's rotated-minus-baseline reward/completion/success on *the same scenes*, then compare with B under the matching permutation. Do not claim positive motor synergy from a drop alone. This demonstrates whether the **learned recipient assignment is functionally used**, not whether modules compensate after impairment.
6. Strong evidence requires repeat across D1 BC seeds 42/43/44, future held-out maps and a temporary perturbation + post-perturbation recovery curve (D3b). Fail/no drop means either routing identity is irrelevant/redundant or the intervention is too weak; do not overgeneralize from a null result.

## After D3a

**D3b**: apply a brief reversible communication/vehicle disturbance at a fixed task-relative trigger, separately measure error during impairment and recovery after restore, and compare to a matching local/physical-feedback-only B control. This moves from loss-of-function to **closed-loop compensation**, the core hypothesis.

**D3c**: repeat on BC seeds43/44 under newly held-out scenes and compare module-specific intervention signatures. Separate representational difference from causal improvement; no skill labels imposed.

Do not launch another PPO engineering investigation as a prerequisite unless D3a directly uncovers a concrete implementation defect.
