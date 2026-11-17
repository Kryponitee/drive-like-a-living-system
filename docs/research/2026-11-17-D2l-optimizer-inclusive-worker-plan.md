# D2l — Optimizer-inclusive MetaDrive worker scaling: 32 versus 64

Date: 2026-11-17. **Preregistered before executing D2l.**

The user approved proceeding with the earlier selected 64-worker configuration. D2k extended sampling-only confirmed 64 workers at 925.4 frames/sec and 32 workers at 551.3 frames/sec on a controlled fixed 24,576-frame run. **That speedup is not necessarily reflected in recurrent PPO training**, because rollout and multiple PPO epochs share the same CPU.

## Hypothesis and sole variable

Compare full official SB3-Contrib 2.6.0 RecurrentPPO training throughput with **32 vs 64 independent CPU MetaDrive workers**. Both use the SAME saved official 300k model/optimizer and VecNormalize state, training algorithm, hyperparameters, device, seed protocol, and **32,768 aggregate training frames**. The only intended configuration difference is number of environment workers.

Fixed `n_steps=128` per worker, `batch_size=128`, `n_epochs=4`, same 259D observations, traffic density 0.1, map3, horizon 800, same SB3 2.6.0 overlay env, same trained LSTM weights. Thus the rollout size per PPO update differs (32×128 vs 64×128) and the number of outer iterations differs (8 vs 4). We keep the same total environment frames and comparable aggregate minibatch-optimizer exposure; this is a **systems timing comparison**, NOT a paired policy-learning comparison or a statistically controlled intervention for learning outcomes.

Two tests execute **sequentially**, each using a fresh model loaded from the same original checkpoint, original saved normalization and separately configured seed ranges. Do not overwrite the 300k source. Time start/first reset, full `learn()` runtime, `model.train()` wall time and optimizer step count; infer rollout+other wall time as `learn_wall - train_wall`. Save each condition immediately and record throughput. No new teacher data, no environment dependency install, and no new validation scenario ranking.

## Gate

Select 64 if its optimizer-inclusive frames/sec exceeds 32 by **at least 10%**, with no numerical or OOM errors and adequate memory. Otherwise retain 32 for formal experiments; this can supersede D2k's sampling-only preference. Record both results even if they contradict the original inference-only speedup.

Avoid treating 32/64 conditions as fair **scientific performance** comparisons, because per-update rollout batch and episode coverage differ, despite equal frames and total expected minibatches.

Data saved only to `/mnt/datasets/lxb/dls/reports/d2l/`; original checkpoints and normalizer untouched. Official reference (random LSTM, only 2/24 arrivals after 300k frames) is for training-system speed, not the A/B/C modular control experiment.
