"""Full batch, normalized behavior observations, and checkpoint round trip.

Run: python -m examples.test_cascade_training_integration
Uses the configured rollout length and physical parameters; all output is
written to temporary directories. Run Step 1 and Step 3 smoke tests first.
"""

import copy
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.utils.configs_tools import get_defaults_yaml_args


def main():
    algo, env = get_defaults_yaml_args("happo", "cascade_reservoir")
    algo["eval"]["use_eval"] = False
    algo["device"]["cuda"] = False
    algo["train"]["n_rollout_threads"] = 1
    algo["eval"]["n_eval_rollout_threads"] = 1
    algo["train"]["model_dir"] = None
    args = {"algo": "happo", "env": "cascade_reservoir", "exp_name": "audit"}
    with tempfile.TemporaryDirectory(prefix="harl-training-audit-") as directory:
        algo["logger"]["log_dir"] = directory
        runner = CascadeReservoirRunner(args, algo, env)
        restored = None
        try:
            runner.warmup()
            runner.prep_rollout()
            for t in range(algo["train"]["episode_length"]):
                collected = runner.collect(t)
                sample = runner._pending_sampling_result
                transition = runner.envs.step(collected[1])
                runner.insert((*transition, *collected))
                for i, buffer in enumerate(runner.actor_buffer):
                    np.testing.assert_array_equal(buffer.obs[t], sample.decision_observations[i])
                    np.testing.assert_array_equal(buffer.available_actions[t], sample.action_masks[i])
                    np.testing.assert_array_equal(buffer.actions[t], sample.actions[:, i])
                    np.testing.assert_array_equal(buffer.action_log_probs[t], sample.action_log_probs[:, i])
                    if i:
                        key = runner.raw_train_env.reservoir_order[i]
                        scale = runner.raw_train_env.action_mappers[key].map_max_release_m3s
                        np.testing.assert_allclose(buffer.obs[t, :, -1] * scale,
                                                   sample.inflows_m3s[:, i], rtol=1e-6)
                    info = transition[4][0][i]
                    assert info["executed_release_m3s"] == info["target_release_m3s"]
                for j in range(runner.constraint_buffer.num_constraints):
                    np.testing.assert_array_equal(runner.critic_buffer.share_obs[t],
                                                  runner.constraint_buffer.get_buffer(j).share_obs[t])
                if (t + 1) % 50 == 0:
                    print(f"rollout {t + 1}", flush=True)
            print("PASS Step 4: behavior and critic indices", flush=True)
            runner.compute()
            buffers = [runner.critic_buffer] + [runner.constraint_buffer.get_buffer(j)
                       for j in range(runner.constraint_buffer.num_constraints)]
            assert all(np.isfinite(buffer.returns).all() for buffer in buffers)
            print("PASS Step 5: compute", flush=True)
            seen, order = {}, []
            for i, actor in enumerate(runner.actor):
                original = actor.train

                def wrapped(*a, _i=i, _original=original, **kw):
                    order.append(_i)
                    result = _original(*a, **kw)
                    seen[_i] = result
                    return result

                actor.train = wrapped
            runner.prep_training()
            with patch("harl.runners.cascade_reservoir_runner.torch.randperm",
                       return_value=torch.tensor([4, 2, 0, 3, 1])):
                infos, critic_info = runner.train()
            assert order == [4, 2, 0, 3, 1]
            assert all(infos[i] is seen[i] for i in range(5))
            assert all(np.isfinite(float(v)) for info in infos + [critic_info]
                       for v in info.values())
            runner.after_update()
            print("PASS Steps 6-7: actual training and metric identities", flush=True)
            runner.save()
            reload_algo = copy.deepcopy(algo)
            reload_algo["train"]["model_dir"] = str(runner.save_dir)
            restored = CascadeReservoirRunner(dict(args, exp_name="reload"), reload_algo, env)
            original_networks = [a.actor for a in runner.actor] + [runner.critic.critic] + [
                c.critic for c in runner.constraint_critic.critics]
            loaded_networks = [a.actor for a in restored.actor] + [restored.critic.critic] + [
                c.critic for c in restored.constraint_critic.critics]
            for original, loaded in zip(original_networks, loaded_networks):
                for name, tensor in original.state_dict().items():
                    assert torch.equal(tensor, loaded.state_dict()[name])
            np.testing.assert_array_equal(runner.lagrangian_manager.snapshot(),
                                          restored.lagrangian_manager.snapshot())
            print("PASS checkpoint encoding, five Actors, six Critics and lambda round trip", flush=True)
        finally:
            if restored is not None:
                restored.close()
            runner.close()


if __name__ == "__main__":
    main()
