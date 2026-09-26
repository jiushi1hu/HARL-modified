"""Step 3 smoke test: collect one joint action and execute it once."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RESERVOIR_ORDER = ("WDD", "BHT", "XLD", "XJB", "THR")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    # Process titles are unrelated to the smoke test. Some local Python
    # environments do not have this optional HARL dependency installed.
    if importlib.util.find_spec("setproctitle") is None:
        sys.modules["setproctitle"] = SimpleNamespace(
            setproctitle=lambda _title: None
        )

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
    from harl.utils.configs_tools import get_defaults_yaml_args

    algo_args, env_args = get_defaults_yaml_args(
        "happo", "cascade_reservoir"
    )
    algo_args["train"]["n_rollout_threads"] = 1
    algo_args["eval"]["n_eval_rollout_threads"] = 1
    algo_args["eval"]["use_eval"] = False
    algo_args["device"]["cuda"] = False
    algo_args["train"]["episode_length"] = 2

    args = {
        "algo": "happo",
        "env": "cascade_reservoir",
        "exp_name": "step3_smoke",
    }

    print("STEP 3 | prepare_step → collect → env.step", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"NumPy: {np.__version__}", flush=True)

    with tempfile.TemporaryDirectory(prefix="harl-step3-") as output_dir:
        algo_args["logger"]["log_dir"] = output_dir
        runner = CascadeReservoirRunner(args, algo_args, env_args)
        try:
            runner.warmup()
            runner.prep_rollout()
            env = runner.raw_train_env
            context = env.prepare_step()
            before_storage = dict(env.storage_m3)
            before_date = env.current_date
            print(
                f"[PASS] prepare_step: {before_date}, "
                f"mode={context['safety_result'].mode}",
                flush=True,
            )

            values, actions, old_log_probs, actor_rnns, reward_rnn = (
                runner.collect(0)
            )
            sampling = runner._pending_sampling_result
            assert sampling is not None
            assert actions.shape == (1, 5, 1)
            assert np.isfinite(values).all()
            assert np.isfinite(old_log_probs).all()
            assert np.isfinite(actor_rnns).all()
            assert np.isfinite(reward_rnn).all()
            assert np.isfinite(runner._pending_constraint_values).all()
            print(
                f"[PASS] collect: actions={actions[0, :, 0].tolist()}, "
                f"constraint_critics={runner.constraint_buffer.num_constraints}",
                flush=True,
            )

            obs, share_obs, rewards, dones, infos, available_actions = (
                runner.envs.step(actions)
            )
            assert obs.shape == (1, 5, 18)
            assert share_obs.shape == (1, 5, 85)
            assert rewards.shape == (1, 5, 1)
            assert dones.shape == (1, 5)
            assert available_actions.shape == (1, 5, 201)
            assert not dones.any()
            assert env.current_date > before_date
            print("[PASS] env.step: returned HARL interface shapes", flush=True)

            step_infos = infos[0]
            assert len(step_infos) == 5
            total_energy = 0.0
            previous_release = None
            for index, reservoir_id in enumerate(RESERVOIR_ORDER):
                info = step_infos[index]
                inflow = float(info["inflow_m3s"])
                release = float(info["executed_release_m3s"])
                next_storage = float(info["next_storage_m3"])
                expected_inflow = (
                    context["external_inflow_m3s"]
                    if index == 0
                    else previous_release
                    + context["interval_inflows_m3s"][reservoir_id]
                )
                expected_storage = before_storage[reservoir_id] + (
                    inflow - release
                ) * env.timestep_seconds
                p1 = env.p1_constraints[reservoir_id]

                assert info["reservoir_id"] == reservoir_id
                assert info["action_index"] == int(actions[0, index, 0])
                assert np.isclose(inflow, expected_inflow, atol=1e-6)
                assert np.isclose(inflow, sampling.inflows_m3s[0, index], atol=1e-6)
                assert np.isclose(
                    next_storage, expected_storage, rtol=0.0, atol=1.0
                )
                assert np.isclose(
                    release, info["target_release_m3s"], atol=1e-9
                )
                assert np.isclose(
                    release, sampling.executed_releases_m3s[0, index], atol=1e-9
                )
                assert p1.release.min_release_m3s <= release <= p1.release.max_release_m3s
                assert (
                    p1.level.min_level_m - 1e-5
                    <= info["next_level_m"]
                    <= p1.level.max_level_m + 1e-5
                )
                assert np.isclose(env.storage_m3[reservoir_id], next_storage)
                assert np.isclose(env.previous_release_m3s[reservoir_id], release)
                assert np.isfinite(info["power_mw"])
                assert np.isfinite(info["energy_mwh"])
                print(
                    f"[PASS] {reservoir_id}: inflow={inflow:.2f}, "
                    f"Q_exec={release:.2f}, Z_next={info['next_level_m']:.3f}",
                    flush=True,
                )
                total_energy += float(info["energy_mwh"])
                previous_release = release

            expected_reward = total_energy / env.reward_reference_mwh
            assert np.allclose(rewards[0, :, 0], expected_reward, atol=1e-7)
            assert all(
                np.isclose(info["system_reward"], expected_reward)
                for info in step_infos
            )
            assert all(
                np.isclose(info["system_energy_mwh"], total_energy)
                for info in step_infos
            )
            print(
                f"[PASS] reward: E_total={total_energy:.3f} MWh, "
                f"reward={expected_reward:.8f}",
                flush=True,
            )

            constraint_ids = tuple(
                env.constraint_cost_evaluator.constraint_ids
            )
            for info in step_infos:
                assert tuple(info["constraint_ids"]) == constraint_ids
                assert info["constraint_costs"].shape == (len(constraint_ids),)
                assert info["constraint_active_flags"].shape == (
                    len(constraint_ids),
                )
                assert np.isfinite(info["constraint_costs"]).all()
                assert np.array_equal(
                    info["constraint_costs"], step_infos[0]["constraint_costs"]
                )
            print(
                f"[PASS] constraint cost: "
                f"{dict(zip(constraint_ids, step_infos[0]['constraint_costs'].tolist()))}",
                flush=True,
            )
            print("STEP 3 PASSED", flush=True)
        finally:
            runner.envs.close()
            if runner.eval_envs is not None:
                runner.eval_envs.close()
            runner.writter.close()


if __name__ == "__main__":
    main()
