# DLS D3c — Three-seed replicated direct-message necessity, frozen BC (preregistered)

Date: 2026-11-17; **written before reading D3c outcomes**.

## One scientific question

D3a (recipient permutation) and D3a2 (norm-matched own-only substitution) under frozen C BC training seed42 did not cause robust driving losses, despite prior D1f fresh-scene data showing C>B at traffic density0.10 for BC training seeds42,43,44 individually. **Is the apparent lack of necessary cross-module information reproducible across independent trained C policies?**

No further PPO engineering is required to ask this question. Do NOT train or change model capacity. Source D1 BC C checkpoints `C-seed42.pt`, `C-seed43.pt`, `C-seed44.pt` must all already exist and have distinct hashes.

## Strict protocol

- Freeze the three distinct original D1 C policies; replicate exactly D3a2's step-code intervention `\tilde aux_i = ||aux_i|| v_i / max(||v_i||,1e-12)`. Compare against unmodified C baseline for every model, **including redoing seed42 on new scenes** to avoid selective reuse of prior outcomes.
- One primary environment density **0.10** (D1f/D2n's originally specified training and evaluation density), maps3, max800 steps, **new scene seeds 32000..32023**, 24 episodes ×2 conditions ×3 training seeds = **144 new episodes**.
- Same exact observation normalization per C checkpoint and original actor source code; input 259D, output deterministic two control actions. No expert queries, no PPO, no optimizer, no new training inputs, no extra model architecture.
- All 6 conditions run as 6 CPU processes; each compares identical scene seeds. Explicitly assert unmodified step parity (<2e-6), finite per-step actions, per-recipient incoming auxiliary message energy difference <1e-3 under own-only replacement, and frozen source checkpoint digest unchanged.
- Report each seed's paired success, collision, off-road, original/replaced mean raw reward, mean completion, per-scene reward deltas, and role of route / message magnitude. No cross-seed pooling as if all scene/episode samples independent.

## Before results: fixed evidence bar

- **Necessary cross-module dependency candidate**: at least 2/3 training seeds show **≥2/24 fewer successes and ≥15 lower average raw reward** after own-only replacement on the same evaluation scenes, without a large opposing gain (>15 raw reward) in the third. This is an exploratory operational threshold, not a hypothesis test.
- **No robust necessity evidence**: fewer than 2/3 seeds meet that joint threshold; especially if own-only is neutral or better in most seeds. This does not prove that communication is impossible to learn in a different setting; it shows current learned C modules have not demonstrated a necessary direct information path.
- Run D3b temporary impairment/return-to-baseline feedback compensation only after reviewing D3c. If no necessary cross-channel use is established, do not automatically continue searching for a message lesion that yields a desired effect. Investigate what learning pressure could create cross-module information demand (scenario coupling/task coverage), a new scientific stage requiring explicit hypothesis.
- Small sample and correlated maps remain; a positive candidate requires further independent seeds and held-out map styles.

All source checkpoints and the D3a/D3a2 files remain unchanged, with new outputs under `/mnt/datasets/lxb/dls/reports/d3c/`.
