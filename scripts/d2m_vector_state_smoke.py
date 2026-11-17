#!/usr/bin/env python3
"""D2m: prove batched asynchronous A/B/C recurrence matches per-agent execution.

Test behavior equivalence, exact phase-state codec, nonzero gradients and
old-policy action probability replay. Uses frozen D1 seed42 BC weights only.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from dls_p0 import OBS_DIM, SLOW_PERIOD, model_info, nparams
from dls_recurrent_ppo import squashed_gaussian_log_probability
from dls_vector_state_adapter import DLSVectorAdapter

ROOT = Path("/mnt/datasets/lxb/dls")
N = 48
STEPS = 37


def main():
    assert os.path.ismount("/mnt/datasets")
    torch.set_num_threads(1)
    torch.manual_seed(835)
    ROOT_OUT = ROOT / "reports/d2m"
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT_OUT / "vector-state-summary.json").exists()
    results = {}
    for variant in "ABC":
        checkpoint = torch.load(
            ROOT / "checkpoints/d1-v1" / f"{variant}-seed42.pt",
            map_location="cpu", weights_only=False,
        )
        actor, _ = model_info(variant)
        assert checkpoint["variant"] == variant
        actor.load_state_dict(checkpoint["state_dict"], strict=True)
        assert nparams(actor) == checkpoint["num_params"]
        actor.eval()

        adapter = DLSVectorAdapter(actor)
        assert adapter.flat_state_size() > 16
        vector_state = adapter.initial_state(N, torch.device("cpu"))
        reference_states = [actor.initial_state(1, torch.device("cpu")) for _ in range(N)]
        reference_phase = [0] * N
        rng = torch.Generator().manual_seed(9012)
        max_action_error = 0.0
        max_hidden_error = 0.0
        max_slow_error = 0.0
        max_replay_ratio_error = 0.0
        phase_diversity = set()
        start = time.monotonic()
        with torch.no_grad():
            for t in range(STEPS):
                obs = torch.randn((N, OBS_DIM), generator=rng) * 0.75
                reset = torch.tensor(
                    [t == 0 or (i * 7 + t * 3) % 29 == 0 for i in range(N)],
                    dtype=torch.bool
                )
                a_vec, next_vec = adapter.step(obs, vector_state, reset)
                assert a_vec.shape == (N, 2)
                assert torch.isfinite(a_vec).all() and a_vec.abs().max() < 0.999
                packed = adapter.flatten(next_vec)
                roundtrip = adapter.unflatten(packed)
                assert torch.equal(roundtrip.phase, next_vec.phase)
                assert torch.equal(roundtrip.hidden, next_vec.hidden)
                assert torch.equal(roundtrip.slow, next_vec.slow)

                a_ref, h_ref, z_ref = [], [], []
                for i in range(N):
                    if bool(reset[i]):
                        reference_states[i] = actor.initial_state(1, torch.device("cpu"))
                        reference_phase[i] = 0
                    action_i, next_i = actor.step(
                        obs[i:i+1], reference_states[i], reference_phase[i]
                    )
                    reference_states[i] = next_i
                    reference_phase[i] = (reference_phase[i] + 1) % SLOW_PERIOD
                    a_ref.append(action_i)
                    h_ref.append(next_i[0])
                    z_ref.append(next_i[1])
                a_ref = torch.cat(a_ref, dim=0)
                h_ref = torch.cat(h_ref, dim=0)
                z_ref = torch.cat(z_ref, dim=0)
                err_a = float((a_vec-a_ref).abs().max())
                err_h = float((next_vec.hidden-h_ref).abs().max())
                err_z = float((next_vec.slow-z_ref).abs().max())
                max_action_error = max(max_action_error,err_a)
                max_hidden_error = max(max_hidden_error,err_h)
                max_slow_error = max(max_slow_error,err_z)
                assert err_a < 2e-5 and err_h < 2e-5 and err_z < 2e-5
                assert next_vec.phase.tolist() == reference_phase
                phase_diversity.update(next_vec.phase.tolist())

                # No policy update: recompute from the very same memory/obs and
                # latent action. The log-probability ratio MUST be 1.
                a_again, next_again = adapter.step(obs, vector_state, reset)
                assert torch.allclose(a_again,a_vec,atol=1e-6,rtol=1e-6)
                mu_old = torch.atanh(a_vec)
                mu_replay = torch.atanh(a_again)
                noise = torch.randn((N,2),generator=rng) * 0.05
                latent = mu_old + noise
                oldlp = squashed_gaussian_log_probability(
                    mu_old, mu_old.new_full((2,),float(np.log(0.05))), latent)
                newlp = squashed_gaussian_log_probability(
                    mu_replay, mu_old.new_full((2,),float(np.log(0.05))), latent)
                error = float((torch.exp(newlp-oldlp)-1).abs().max())
                max_replay_ratio_error = max(max_replay_ratio_error,error)
                assert error < 3e-5
                vector_state=next_vec

        assert phase_diversity == set(range(SLOW_PERIOD)), phase_diversity
        assert float(vector_state.phase.float().std()) > 0.2

        # Differentiable tensor regrouping MUST preserve gradients for true PPO
        # time-unrolled recurrent learning, not only no-grad inference.
        for parameter in actor.parameters():
            parameter.grad = None
        differentiable_state = adapter.initial_state(8,torch.device("cpu"))
        objective = torch.zeros((), dtype=torch.float32)
        for t in range(10):
            x=torch.randn((8,OBS_DIM),generator=rng)
            starts=torch.tensor([t==0 or (i*3+t)%11==0 for i in range(8)],dtype=torch.bool)
            actions,differentiable_state=adapter.step(x,differentiable_state,starts)
            objective=objective+actions.square().mean()
        objective.backward()
        grad_sum=sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
        assert grad_sum > 0 and np.isfinite(grad_sum),f"{variant}: missing recurrent parameter gradients"
        assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
        actor.zero_grad(set_to_none=True)

        record={
            "variant":variant,"frozen_BC_seed":42,
            "original_actor_trainable_params":nparams(actor),
            "num_async_envs_tested":N,"ticks":STEPS,
            "slow_period":SLOW_PERIOD,
            "phases_covered":sorted(phase_diversity),
            "flat_state_size":adapter.flat_state_size(),
            "max_action_error_batched_vs_single":max_action_error,
            "max_hidden_error_batched_vs_single":max_hidden_error,
            "max_slow_error_batched_vs_single":max_slow_error,
            "max_old_policy_ratio_replay_error":max_replay_ratio_error,
            "differentiable_actor_grad_norm_L1":grad_sum,
            "wall_seconds":time.monotonic()-start,
            "actor_weights_unchanged":True,
            "test_uses_synthetic_observation_vectors":True,
        }
        out=ROOT_OUT/f"vector-state-{variant}.json"
        assert not out.exists()
        out.write_text(json.dumps(record,indent=2),encoding="utf-8")
        results[variant]=record
        print(f"PASS D2m {variant}: batched/single max_action_err={max_action_error:.2e}, "
              f"state_err={max(max_hidden_error,max_slow_error):.2e}, "
              f"logprob_ratio_err={max_replay_ratio_error:.2e}; grad_L1={grad_sum:.3f}",flush=True)
    assert results["B"]["original_actor_trainable_params"] == results["C"]["original_actor_trainable_params"]
    assert results["B"]["flat_state_size"] == results["C"]["flat_state_size"]
    result={
        "stage":"D2m controller-specific asynchronous vectorized state clock adapter",
        "models":results,
        "passed_actor_states_and_probabilities":True,
        "does_not_establish_Sb3_native_custom_policy_integration":True,
        "next_required_action":"Implement and test custom recurrent policy adapter for official training buffer, or move to mature TorchRL PPO components; do not replace A/B/C actors.",
        "test_type":"CPU-only deterministic replay, no MetaDrive rollout or actor training",
    }
    out=ROOT_OUT/"vector-state-summary.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("D2m asynchronous controller state replay PASS",flush=True)


if __name__=="__main__":
    main()
