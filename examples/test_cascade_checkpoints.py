"""Checkpoint lifecycle regression using real short rollouts and optimizers."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.utils.configs_tools import get_defaults_yaml_args


class CheckpointTests(unittest.TestCase):
    def make_runner(self, directory, **train):
        algo, env = get_defaults_yaml_args("happo", "cascade_reservoir")
        algo["train"].update(episode_length=4, num_env_steps=4, model_dir=None,
                             eval_interval=50, save_interval=50)
        algo["train"].update(train)
        algo["eval"]["use_eval"] = False
        algo["device"]["cuda"] = False
        algo["logger"]["log_dir"] = tempfile.mkdtemp(dir=directory)
        return CascadeReservoirRunner(
            dict(algo="happo", env="cascade_reservoir", exp_name="checkpoint_test"),
            algo, env)

    @staticmethod
    def networks(runner):
        return [a.actor for a in runner.actor] + [runner.critic.critic] + [
            c.critic for c in runner.constraint_critic.critics]

    def test_short_run_saves_initial_and_final_and_restores_all_components(self):
        with tempfile.TemporaryDirectory(prefix="harl-checkpoints-") as directory:
            runner = self.make_runner(directory)
            restored = None
            try:
                initial = [copy.deepcopy(a.actor.state_dict()) for a in runner.actor]
                runner.run()  # 1 update, less than save_interval=50; eval disabled.
                root = Path(runner.save_dir)
                for folder, kind, updates in [(root / "initial", "run_start", 0),
                                               (root, "final", 1)]:
                    metadata = json.loads((folder / "checkpoint_metadata.json").read_text())
                    self.assertEqual(metadata["kind"], kind)
                    self.assertEqual(metadata["completed_updates_in_run"], updates)
                    self.assertEqual(metadata["training_steps_in_run"], 4 * updates)
                    self.assertFalse(metadata["exact_training_resume"])
                    self.assertTrue((folder / "config.json").is_file())
                    CascadeReservoirRunner._validate_observation_checkpoint(folder)
                changed = False
                for i, actor in enumerate(runner.actor):
                    baseline = torch.load(root / "initial" / f"actor_agent{i}.pt")
                    for key, value in initial[i].items():
                        self.assertTrue(torch.equal(value, baseline[key]))
                        changed |= not torch.equal(value, actor.actor.state_dict()[key])
                self.assertTrue(changed, "real training must distinguish final from initial")
                restored = self.make_runner(directory, model_dir=str(root))
                for source, loaded in zip(self.networks(runner), self.networks(restored)):
                    for key, value in source.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
                for source, loaded in [(runner.value_normalizer, restored.value_normalizer),
                        *zip(runner.constraint_critic.value_normalizers,
                             restored.constraint_critic.value_normalizers)]:
                    for key, value in source.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
                np.testing.assert_array_equal(runner.lagrangian_manager.snapshot(),
                                              restored.lagrangian_manager.snapshot())
            finally:
                if restored is not None:
                    restored.close()
                runner.close()

    def test_periodic_saving_is_independent_of_eval_and_final_is_not_duplicated(self):
        with tempfile.TemporaryDirectory(prefix="harl-checkpoints-") as directory:
            runner = self.make_runner(directory, num_env_steps=8, save_interval=1)
            try:
                with patch.object(runner, "_save_training_checkpoint",
                                  wraps=runner._save_training_checkpoint) as save:
                    runner.run()
                self.assertEqual([(c.args[0], c.kwargs) for c in save.call_args_list],
                                 [(0, {"initial": True}), (1, {"final": False}),
                                  (2, {"final": True})])
            finally:
                runner.close()

    def test_final_is_saved_before_evaluation_failure(self):
        with tempfile.TemporaryDirectory(prefix="harl-checkpoints-") as directory:
            runner = self.make_runner(directory, eval_interval=1)
            try:
                runner.algo_args["eval"]["use_eval"] = True
                with patch.object(runner, "eval", side_effect=RuntimeError("evaluation failed")):
                    with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                        runner.run()
                metadata = json.loads((Path(runner.save_dir) / "checkpoint_metadata.json").read_text())
                self.assertEqual(metadata["kind"], "final")
                self.assertEqual(metadata["completed_updates_in_run"], 1)
            finally:
                runner.algo_args["eval"]["use_eval"] = False
                runner.close()

    def test_failed_save_restores_destination_and_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory(prefix="harl-checkpoints-") as directory:
            runner = self.make_runner(directory)
            try:
                original = runner.save_dir
                with patch.object(runner, "save", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        runner._save_training_checkpoint(0, initial=True)
                self.assertEqual(runner.save_dir, original)
                self.assertFalse((Path(original) / "initial" / "checkpoint_metadata.json").exists())
            finally:
                runner.close()

    def test_old_config_fallback_and_invalid_intervals(self):
        algo, env = get_defaults_yaml_args("happo", "cascade_reservoir")
        args = dict(algo="happo", env="cascade_reservoir")
        del algo["train"]["save_interval"]
        CascadeReservoirRunner._validate_configuration(args, algo, env)
        for value in (0, -1, True, 1.5):
            algo["train"]["save_interval"] = value
            with self.assertRaisesRegex(ValueError, "save_interval"):
                CascadeReservoirRunner._validate_configuration(args, algo, env)


if __name__ == "__main__":
    unittest.main()
