"""Regression checks for hydraulic identity and randomized HAPPO updates.

Run from the repository root with:
    python -m unittest examples.test_cascade_training_order
The small arrays below are test fixtures, not physical configuration.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from harl.algorithms.lagrangian_manager import LagrangianManager
from harl.common.buffers.constraint_buffer import ConstraintBatchStatistics
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.envs.cascade_reservoir.sequential_sampler import SequentialSampler


class TrainingOrderTest(unittest.TestCase):
    def test_downstream_inflow_scale_preserves_base_features_and_physical_flow(self):
        # Synthetic feature arrays only; no production parameters are changed.
        base = np.linspace(0., 1., 34, dtype=np.float32).reshape(2, 17)
        flows = np.array([10000., 120000.])
        kwargs = dict(base_observation=base, current_inflow_m3s=flows,
                      inflow_scale_m3s=100000.)
        obs = SequentialSampler._build_decision_observation(
            reservoir_id="THR", **kwargs)
        self.assertEqual(obs.shape, (2, 18))
        np.testing.assert_array_equal(obs[:, :17], base)
        np.testing.assert_allclose(obs[:, -1], [0.1, 1.2])
        np.testing.assert_allclose(obs[:, -1] * 100000., flows)
        np.testing.assert_array_equal(flows, [10000., 120000.])
        wdd = SequentialSampler._build_decision_observation(
            reservoir_id="WDD", **kwargs)
        np.testing.assert_array_equal(wdd, base)
        self.assertFalse(np.shares_memory(wdd, base))
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SequentialSampler._build_decision_observation(
                    reservoir_id="THR", base_observation=base,
                    current_inflow_m3s=flows, inflow_scale_m3s=invalid)

    def test_legacy_or_wrong_observation_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            check = CascadeReservoirRunner._validate_observation_checkpoint
            with self.assertRaisesRegex(ValueError, "Legacy cascade"):
                check(directory)
            path = Path(directory) / "actor_observation_encoding.json"
            path.write_text(json.dumps({"encoding": "raw_inflow"}))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                check(directory)
            path.write_text(json.dumps({
                "encoding": "total_inflow_divided_by_map_max_release_v1"}))
            check(directory)

    def test_metrics_keep_reservoir_identity_and_lambda_stays_frozen(self):
        runner = CascadeReservoirRunner.__new__(CascadeReservoirRunner)
        runner.num_agents = 5
        runner.fixed_order = False
        runner.action_aggregation = "prod"
        runner.algo_args = {"train": {"episode_length": 2, "n_rollout_threads": 1}}
        runner.value_normalizer = None
        runner.critic_buffer = SimpleNamespace(
            returns=np.array([3., 4., 0.], dtype=np.float32).reshape(3, 1, 1),
            value_preds=np.zeros((3, 1, 1), dtype=np.float32),
        )
        runner.constraint_ids = ("test_cost",)
        manager = LagrangianManager(runner.constraint_ids, [0.02], [0.05], [0.5])
        runner.lagrangian_manager = manager
        snapshot = manager.snapshot()
        expected_advantages = np.array([2., 3.], dtype=np.float32).reshape(2, 1, 1)
        events = []

        def record(name):
            np.testing.assert_array_equal(manager.snapshot(), snapshot)
            events.append(name)

        def make_actor(agent_id):
            def train(buffer, advantages, state_type):
                record(agent_id)
                self.assertEqual(state_type, "EP")
                np.testing.assert_allclose(advantages, expected_advantages)
                return {"policy_loss": float(agent_id)}

            return SimpleNamespace(
                train=train,
                evaluate_actions=lambda *args: (torch.zeros(2, 1), None, None),
            )

        runner.actor = [make_actor(i) for i in range(5)]
        runner.actor_buffer = [SimpleNamespace(
            obs=np.zeros((3, 1, 17 if i == 0 else 18), dtype=np.float32),
            available_actions=np.ones((3, 1, 2), dtype=np.float32),
            rnn_states=np.zeros((3, 1, 1, 2), dtype=np.float32),
            actions=np.zeros((2, 1, 1), dtype=np.float32),
            masks=np.ones((3, 1, 1), dtype=np.float32),
            active_masks=np.ones((3, 1, 1), dtype=np.float32),
            update_factor=lambda factor: None,
        ) for i in range(5)]

        def reward_train(*args):
            record("reward")
            return {}

        def constraint_train(*args):
            record("constraint")
            return {}

        runner.critic = SimpleNamespace(train=reward_train)
        runner.constraint_critic = SimpleNamespace(
            compute_advantages=lambda buffer: np.full((2, 1, 1), 2., dtype=np.float32),
            train=constraint_train,
        )
        statistics = ConstraintBatchStatistics(
            constraint_ids=runner.constraint_ids,
            discounted_active_means=np.array([0.12], dtype=np.float32),
            active_weight_sums=np.array([1.99], dtype=np.float32),
            active_counts=np.array([2]),
            valid_flags=np.array([True]),
            budgets=manager.budgets,
        )
        runner.constraint_buffer = SimpleNamespace(
            compute_discounted_active_statistics=lambda: statistics,
        )
        update_order = [4, 2, 0, 3, 1]
        with patch("harl.runners.cascade_reservoir_runner.torch.randperm",
                   return_value=torch.tensor(update_order)):
            actor_info, _ = runner.train()

        self.assertEqual(events, update_order + ["reward", "constraint"])
        np.testing.assert_allclose(manager.snapshot(), [0.505])
        self.assertEqual([info["policy_loss"] for info in actor_info],
                         [0., 1., 2., 3., 4.])


if __name__ == "__main__":
    unittest.main()
