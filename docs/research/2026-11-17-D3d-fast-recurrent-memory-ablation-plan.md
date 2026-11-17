# DLS D3d — Frozen C recurrent-state necessity under original sensor observations

2026-11-17. Pre-registered **before** D3d; follows D3c result where C direct-message dependence varies among original trained BC seed42/43/44.

## One point

**Is C's learned control dependent on retained recurrent hidden state h at all, even when direct cross-module aux messages are not consistently necessary?** All C modules see the same current 259-dimensional observation and shared slow-state z. Thus h persistence may not be needed for the BC target (possibly nearly reactive expert).

## Intervention and controls

- Original frozen D1 BC C actor weights seed42/43/44, same map3 and traffic density0.10, D3c scene seeds 32000..32023.
- Every control step before the original actor.step, set **only h_{t-1} (all 4×32 dimensions) to zero**; keep slow-state z untouched, keep the same step counter t and the original slow-period5 update rule. This is a test of retained *fast recurrent memory*, not of whether direct communication or all temporal context matter, as z and observation remain.
- Reuse D3c per-seed original baseline JSONs from data disk. Do not re-simulate old baseline or train any model. Three new conditions ×24 episodes =72 episodes, in parallel CPU processes. Exact same D1 checkpoint and normalization, no PPO, no new teacher data.
- Verify no action/states differ for the first step (before any history exists), and original checkpoint SHA256 unchanged.
- Compare to D3c's own-only message substitution for each seed on the **same scenes** to distinguish two different lesion types: (i) no immediate cross-message content but intact own temporal state, (ii) erase all own temporal state. Only raw reward, success/collision/road departure, route completion and per-scene paired differences are evaluated.
- **A priori qualitative decision**: If recurrent reset sharply worsens success and reward for 2/3 seeds, while D3c own-only does not do so consistently, own temporal memory is a stronger candidate mechanism than intermodule communication. If recurrent reset is neutral, current BC controller may be largely reactive. A decline cannot prove temporal information is intrinsically required because resetting h is out-of-distribution.
- This is a descriptive diagnostic using reused development scenes, **not a final confirmatory causal-compensation experiment**. A claim of feedback compensation still needs temporary impairment, release and recovery metrics on new scenes.

No architecture change, no changes to P0 scientific module definitions. D3d should decide whether to test actual module memory dependence or investigate richer partial-observation tasks before further closed-loop training.
