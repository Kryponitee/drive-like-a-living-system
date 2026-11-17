# DLS P0 — Official RL stack audit before more preparation work (2026-11-17)

## Motivation

Recent D2b–D2g experiments used a small, locally implemented recurrent PPO path with **4,096 environment transitions per architecture**. Critics explain only ~0.8–1.9% of observed discounted-return variance. That result is NOT evidence that the A/B/C architectures cannot learn or that PPO is fundamentally unsuitable. First audit established upstream implementations and actual simulator protocol, rather than inventing new standard training infrastructure.

## Official sources consulted

1. **MetaDrive official [Training](https://metadrive-simulator.readthedocs.io/en/latest/training.html)**: example uses Stable-Baselines3 PPO + `SubprocVecEnv` and `total_timesteps=300_000`. This is an **illustrative example budget**, not a requirement or directly equivalent benchmark. Environment must be one MetaDrive instance **per OS process** per [Getting Started](https://metadrive-simulator.readthedocs.io/en/latest/get_start.html). The public documentation page currently bears an older MetaDrive version label; the DLS server uses 0.4.3. Confirm API semantics against installed release before claiming exact equivalence.
2. **[SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)**: ready-made recurrent PPO with LSTM policies; manages recurrent state and episode starts. This is the most straightforward **verified training baseline** on MetaDrive. It is NOT a drop-in trainer for the bespoke A/B/C controllers: their GRU/4-module, slow-loop state format is not the library's default LSTM policy state. Replacing those models with `MlpLstmPolicy` would change the scientific variable.
3. **[PyTorch TorchRL PPO](https://docs.pytorch.org/rl/main/tutorials/coding_ppo.html)**: provides standardized `ClipPPOLoss`, `GAE`, `TanhNormal`, collectors, TensorDict storage. [Recurrent module documentation](https://docs.pytorch.org/rl/main/reference/modules_rnn.html) describes explicit state fields and `is_init` resets. More flexible for integrating A/B/C with custom state, but needs a tested adapter to carry their nonstandard recurrent state and time phase. Avoid claiming this is plug-and-play.
4. **[TorchRL ValueNorm](https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.ValueNorm.html)** / [PopArtValueNorm](https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.PopArtValueNorm.html): mature target-scale handling for value heads. Critic normalization should be reviewed rather than hardcoding new schemes without references. Ensure normalized values are **denormalized** for GAE when appropriate.
5. **[CleanRL PPO documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/)** and [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): implementation checklists include parallel rollouts, time-limit bootstrap, advantage normalization, clipped value loss, optimizer settings, KL/clip fractions, reward and observation scaling. CleanRL's provided continuous-action PPO and Atari-LSTM scripts are different examples; don't misrepresent their existence as a drop-in **continuous recurrent** policy.

## Findings in our own preliminary implementation

- Prior 4,096-step tests are **smoke diagnostics**, not comparable to MetaDrive's official 300k-step PPO training example; collecting ~one tenth of a percent of that sample count is insufficient for claims of convergence or architecture inadequacy.
- Continuous steering exploration with `sigma~0.27` was empirically destructive (D2c); `sigma=0.05` reduced the effect.
- Frozen-BC functional anchoring slowed actor drift (D2f), but did not consistently improve closed-loop success. This is a useful **diagnostic**, not a recommended standard PPO component.
- Values from an unnormalized, randomly initialized, low-budget Critic had much smaller variance than realized discounted returns. That makes value-target scaling, critic update capacity, and return distribution relevant. Negative / near-zero EV alone is NOT proof that incorrect advantage directions caused the observed failure.
- The existing local path has **not** been cross-validated against SB3-Contrib or TorchRL on equal data and model, so do not designate it production-ready or a scientific control.
- Truncation vs terminal handling deserves explicit verification. For **D2h** we avoid ambiguous bootstrap labels by excluding entire horizon-censored episodes from the supervised target and independently counting them.

## Chosen next actions / go-no-go

**D2h value-only sanity diagnostic** (not a new RL method):

- Freeze BC actors and reuse existing D1 seed42 parameters. Collect *complete episodes*, at `map=3`, density0.1, `sigma=0.05`, separate scenario seeds train `4600..4623`, validation `4700..4715`. Do not mix source or target windows across splits.
- Compute actual `gamma=0.99` realized returns on episodes with true termination; retain but **exclude time-limit-censored** episodes from supervised training/evaluation. This is valid under true termination; acknowledge selection bias.
- Train **only** the D2f critic architecture via standard PyTorch MSE regression, initialized from D2f checkpoint, fixed 50 epochs, independent held-out evaluation; compare (a) original D2f critic, (b) constant train-mean baseline, (c) fitted critic, report EV, RMSE, MAE, Pearson and episode-level metrics. Do not tune based on validation.
- This only answers: **is the critic at least capable of representing return variation under fixed BC occupancy with ordinary supervised training?** It does not validate PPO or establish generalized value quality.

**Before any more custom PPO learning**:
- Probe available installed `stable_baselines3`, `sb3_contrib`, `torchrl` packages and versions read-only; do not silently install or modify the shared Conda env.
- Choose an official SB3-Contrib `RecurrentPPO` MetaDrive baseline with its own LSTM policy as a **training-system sanity control**, then a separately validated A/B/C-compatible PPO path (prefer wrapping our architectures in mature TorchRL components if their version supports the needed state semantics).
- Log and compare total environmental frames, policy/value updates, reward normalization, entropy/exploration, value targets, hidden-state resets, real terminal vs artificial time limit, same evaluation seeds. All outputs on `/mnt/datasets/lxb/dls`.
- Do not replace the scientific A/B/C model family or interpret results from mismatched recurrent architecture as a matched ablation.

**Research boundary**: D0 and D1 made the environment and basic BC control usable; D2 is still preparation. Only D3 direct-message intervention and matched controls can establish causal modular coordination.
