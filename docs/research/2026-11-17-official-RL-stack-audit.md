# DLS P0 — Official RL stack audit before more preparation work (2026-11-17)

## Motivation

Recent D2b–D2g experiments used a small, locally implemented recurrent PPO path with **4,096 environment transitions per architecture**. Critics explain only ~0.8–1.9% of observed discounted-return variance. That result is NOT evidence that the A/B/C architectures cannot learn or that PPO is fundamentally unsuitable. First audit established upstream implementations and actual simulator protocol, rather than inventing new standard training infrastructure.

## Official sources consulted

1. **MetaDrive official [Training](https://metadrive-simulator.readthedocs.io/en/latest/training.html)**: example uses Stable-Baselines3 PPO + `SubprocVecEnv` and `total_timesteps=300_000`. This is an **illustrative example budget**, not a requirement or directly equivalent benchmark. Its example uses **discrete steering/throttle, traffic_density=0, map='C' and horizon=500**; DLS instead evaluates a continuous 2-dimensional action policy, traffic_density=0.10 and map=3. Matching steps does not make these tasks comparable. Environment must be one MetaDrive instance **per OS process** per [Getting Started](https://metadrive-simulator.readthedocs.io/en/latest/get_start.html). The public documentation page currently bears an older MetaDrive version label; the DLS server uses 0.4.3. Confirm API semantics against installed release before claiming exact equivalence.
2. **[SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)**: ready-made recurrent PPO with LSTM policies; manages recurrent state and episode starts. This is the most straightforward **verified training baseline** on MetaDrive. It is NOT a drop-in trainer for the bespoke A/B/C controllers: their GRU/4-module, slow-loop state format is not the library's default LSTM policy state. Replacing those models with `MlpLstmPolicy` would change the scientific variable.
3. **[PyTorch TorchRL PPO](https://docs.pytorch.org/rl/main/tutorials/coding_ppo.html)**: provides standardized `ClipPPOLoss`, `GAE`, `TanhNormal`, collectors, TensorDict storage. [Recurrent module documentation](https://docs.pytorch.org/rl/main/reference/modules_rnn.html) describes explicit state fields and `is_init` resets. More flexible for integrating A/B/C with custom state, but needs a tested adapter to carry their nonstandard recurrent state and time phase. Avoid claiming this is plug-and-play.
4. **[TorchRL ValueNorm](https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.ValueNorm.html)** / [PopArtValueNorm](https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.PopArtValueNorm.html): mature target-scale handling for value heads. Critic normalization should be reviewed rather than hardcoding new schemes without references. Ensure normalized values are **denormalized** for GAE when appropriate.
5. **[CleanRL PPO documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/)** and [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): implementation checklists include parallel rollouts, time-limit bootstrap, advantage normalization, clipped value loss, optimizer settings, KL/clip fractions, reward and observation scaling. CleanRL's provided continuous-action PPO and Atari-LSTM scripts are different examples; don't misrepresent their existence as a drop-in **continuous recurrent** policy.

## Findings in our own preliminary implementation

- Prior 4,096-step tests are **smoke diagnostics**, not comparable to MetaDrive's official 300k-step PPO training example; collecting **4,096 / 300,000 = 1.37%** of that example's environment-step budget is insufficient for claims of convergence or architecture inadequacy.
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


## 2026-11-17 precision revision: D2h actual result and framework compatibility

**D2h completed with SUCCESS**: https://github.com/Kryponitee/drive-like-a-living-system/actions/runs/44023029712. Both actors and old critic initialization were frozen until the critic-only supervised phase; never update a driving actor. Train scenarios 4600..4623, validation 4700..4715, new compared models share the same frozen-BC policy's trajectories; 50 fixed supervised epochs, raw reward-scale discounted return labels. Excluded time-limit-censored entire episodes.

| Actor | Train uncensored steps | Val uncensored steps | Initial critic EV | Supervised critic EV | Supervised critic RMSE | Constant train-mean RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A | 7918 | 5723 | 0.0228 | **0.2145** | **22.79** | 23.77 |
| B | 8835 | 6486 | 0.0109 | **0.1617** | **23.76** | 24.75 |
| C | 8297 | 5867 | 0.0135 | **-0.1548** | 30.07 | **25.18** |

- **Conclusion is conditional**: A/B critics learned some predictive variation, but only marginally beat a constant-MSE baseline; C's fitted value model underperforms that baseline. None of these critics is a tested replacement in on-policy PPO.
- **Distribution mismatch**: the old D2f critic weights were trained under a different (updated and stochastic) actor; D2h labels came from **frozen original BC** with sigma=0.05. Thus the old-vs-new gap includes state-distribution shift and should not be attributed solely to supervision or value-target normalization.
- **Effective sample unit**: although thousands of labeled steps are stored, there are only 24/16 *episodes* per actor in train/validation; within-episode target correlation and censoring-selection bias make uncertainty much larger than step counts suggest.
- **State/critic mismatch**: the actor A/B/C maintains recurrent hidden state, but current `IdenticalCritic` estimates `V(x_t)` from the instantaneous 259-dimensional observation only. For a history-dependent policy, the relevant value may depend on `(x_t,h_t)`. The near-zero explained variance may therefore be due to *insufficient critic information*, not just training steps or target scaling. SB3-Contrib's standard RecurrentPPO exposes an LSTM critic option; use it as a reference implementation concept, not proof that history solves our specific tasks.

### Server package audit — actually measured

D2h read-only preflight reported:

| Package | Installed in `dls-p0` |
|---|---|
| PyTorch | **2.5.1** |
| Gymnasium | **1.4.0** |
| MetaDrive | **0.4.3** |
| Stable-Baselines3 / SB3-Contrib | **No** |
| TorchRL / TensorDict | **No** |

**Version compatibility is a real blocker to an unpinned `pip install`:**

- Newer SB3 **2.9.x** requires `torch>=2.8` and supports `gymnasium<2`; it cannot be dropped into the *existing torch=2.5.1 stack without changing that dependency*. Official source: https://github.com/DLR-RM/stable-baselines3/releases and https://stable-baselines3.readthedocs.io/en/master/guide/install.html.
- Older SB3 **2.7.x** supports PyTorch>=2.3 but uses Gymnasium range **<1.3.0**; the current Gymnasium 1.4.0 therefore also needs isolation or version adjustment. Official source: https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/install.html and official v2.9 change notes.
- Current TorchRL releases target a substantially newer PyTorch 2.11+ (latest source: https://github.com/pytorch/rl/releases); exact TorchRL+TensorDict versions compatible with 2.5.1 must be selected and smoke-tested, not guessed.
- **Do not install any of these directly into `/home/ps/miniconda3/envs/dls-p0`** or upgrade system PyTorch/CUDA implicitly. First use dependency resolution/dry-run and a size check, then a separately isolated reference environment on **system/root disk**, while caching packages and placing data/model artifacts on `/mnt/datasets/lxb/dls/`. Main env is immutable for fair D1/D2 comparisons.

### Refined gate / scientific priority

1. **No more PPO lambda/sigma sweeps.** D2c/d/e/f/g/h have already isolated exploration-scale effects, actor function drift, and insufficient Critic calibration.
2. **First one-point mechanism test (D2i):** on the **existing D2h full-episode train/validation rollout data**, compare an observation-only critic with a **history-conditioned recurrent critic**, keeping the original BC policy and label sets frozen. Control critic trainable parameter count (target <=2% mismatch), optimization and evaluation budgets, time-limit masks, and episode-level splits. Never claim improved actor control from a value-only regressor. This directly checks whether history changes learnability, instead of randomly modifying the PPO optimizer.
3. **Official baseline engineering gate:** independently install one **version-compatible** SB3-Contrib RecurrentPPO reference in an isolated environment. Check MetaDrive 0.4.3 Gymnasium reset/step, episode_start, time-limit terminal observations, continuous Box action, process-based vec env, exploration and reward conventions on exactly the DLS environment. It is an **engineering oracle, not B/C matched ablation**, because its LSTM architecture differs. Only use performance to validate training infrastructure.
4. **Custom policy integration:** for A/B/C use a well-tested PPO loss/GAE/collector pathway, preferably TorchRL if a compatible release and explicit hidden-state adapter passes tests, otherwise tightly verify a minimal adapter against SB3 calculations. All variants must use the **same** value model and fair compute/rollout budgets.
5. **Preparation exit threshold:** demonstrate recurrent state reset, correct bootstrap and old-policy log-prob replay; reproducible improved **evaluation success/reward on independent seeds** in >=2 training seeds of the reference and at least one custom controller; check that B/C differ only in intended communication topology and quantify FLOPs. If these are not met, preparation is not done. If met, move to D3 **frozen-policy communication intervention and matched local-state perturbation** without further expanding the PPO engineering branch.

**No outcome is to be selected post hoc.** Values above are diagnostic; architecture/mechanism evidence remains unestablished.
