# DLS D3e — Original C policies as strictly stateless current-observation mappings

2026-11-17. Preregistered before running. Follows D3c cross-message source intervention and D3d fast-GRU memory reset.

## Scientific motivation

Frozen C BC seed42/43/44 models did not exhibit consistently necessary direct intermodule messages (D3c), and did not exhibit consistently necessary *fast GRU memory* when the slow-state was preserved (D3d).

The official MetaDrive [Observation documentation](https://metadrive-simulator.readthedocs.io/en/latest/obs.html) states default 259D `LidarStateObservation` comprises ego speed/heading, navigation and nearby obstacle Lidar measurements. Such an observation can provide much of the state needed for basic steering and throttle without integrating hidden-state history. The official documentation page reports an older version label; the installed MetaDrive 0.4.3 has been empirically checked to output 259D in D0/D1, but the exact field semantics should be audited against installed code before creating any partial-observation variant.

## One-point intervention, an explicit stop/go scientific test

Using the **same learned C recurrent policy weights without retraining**, evaluate the *stateless mapping*

`a_t = C_theta.step(o_t, initial_state=(h=0,z=0), t=0).action`,

where original C's slow network recomputes `z` from *current observation* every tick; no hidden/slow state crosses action boundaries. This is a diagnostic extraction of a feedforward map from the existing policy, not a separately trained feedforward control architecture.

The only additional conceptual intervention relative to D3d is removal of retained **slow z**, since fast h was already reset. Only current observation can influence current control. All weights, action decoder, routing and module structure remain identical.

- Frozen original D1 BC C seed42/43/44 (verified distinct checkpoints), MetaDrive map3, density0.10, 24 **paired existing D3c/D3d development scenes 32000..32023**, horizon800, deterministic action.
- Only 3 new simulator conditions ×24 episodes=**72 new episodes**, 3 CPU processes; directly reuse D3c original and D3d fast-h-only-reset outputs, and D3c own-message results.
- At episode start, the original policy and stateless mapping must output identical first action (same zero memory and slow refresh at t=0). Apply a constant t=0 for each subsequent policy call so original `encode` recomputes z each tick, and discard all h,z across ticks; no instrumented model copy is needed.
- Evaluate per-seed raw episode reward, success/collision/off-road, completion and paired per-scene differences, unchanged checkpoint hashes. Do not aggregate repeated scenes across trained models as independent observations.

## Scientific stop/go interpretation

- If stateless control maintains roughly original success and reward in most or all trained seeds, then current learned policy's apparent recurrent/module structure may be functionally unnecessary for the **existing basic-driving BC task**; start redesigning the *task's observable temporal coupling* before claiming emergence of recurrent motor coordination.
- If stateless control collapses but D3d fast-only reset did not, then slowly held z (not fast recurrent h) is a candidate necessary temporal mechanism. Next use a short reversible intervention on z with a matched control, not another generic PPO rebuild.
- A stateless reset is out-of-distribution relative to training; worse performance cannot prove real-world task intrinsically requires memory. A properly trained capacity-matched FF MLP under the **same expert data** would be a stronger second-level necessity control if this test is ambiguous.
- Do not conclude anything about biological motor coordination from a stateless network comparison alone.

No extra BC/PPO training, no new expert dataset, no changes to core A/B/C source. Output goes to `/mnt/datasets/lxb/dls/reports/d3e/`.
