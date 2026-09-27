"""Frozen evaluation tests; short dates are test fixtures only."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.evaluate_cascade import compare, cost_statistics, read_checkpoint
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.utils.configs_tools import get_defaults_yaml_args


class EvaluationTests(unittest.TestCase):
    def test_active_and_discounted_statistics(self):
        rows = [dict(cost='0.2', active='1'), dict(cost='0', active='0'),
                dict(cost='0.6', active='1')]
        result = cost_statistics(rows, .5)
        self.assertAlmostEqual(result['mean_cost'], .4)
        self.assertAlmostEqual(result['discounted_mean_cost'], .28)
        self.assertEqual(result['active_days'], 2)
        self.assertEqual(result['violation_days'], 2)
        empty = cost_statistics([dict(cost='0', active='0')], .99)
        self.assertIsNone(empty['mean_cost'])
        self.assertIsNone(empty['discounted_mean_cost'])

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_checkpoint(directory)

    def test_real_complete_period_without_training_and_config_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='harl-evaluation-') as directory:
            algo, env = get_defaults_yaml_args('happo', 'cascade_reservoir')
            algo['train'].update(episode_length=2, num_env_steps=2, model_dir=None)
            algo['eval']['use_eval'] = False
            algo['device']['cuda'] = False
            algo['logger']['log_dir'] = str(Path(directory) / 'training')
            env['simulation']['end_date'] = '2023-01-03'
            # DataLoader requires an exact date range; isolate a matching fixture.
            fixture = Path(directory) / 'data'
            shutil.copytree(env['data']['root'], fixture)
            inflow = fixture / env['data']['daily_inflow']
            inflow.write_text('\n'.join(inflow.read_text().splitlines()[:4]) + '\n')
            env['data']['root'] = str(fixture)
            runner = CascadeReservoirRunner(
                dict(algo='happo', env='cascade_reservoir', exp_name='evaluation_fixture'), algo, env)
            try:
                runner.run()
                model = Path(runner.save_dir)
            finally:
                runner.close()
            with patch.object(CascadeReservoirRunner, 'run', side_effect=AssertionError('run called')), \
                 patch.object(CascadeReservoirRunner, 'train', side_effect=AssertionError('train called')), \
                 patch.object(CascadeReservoirRunner, 'compute', side_effect=AssertionError('compute called')):
                results = compare(model, model / 'initial', Path(directory) / 'evaluation')
            for label in ('baseline', 'trained'):
                self.assertEqual(results[label]['days'], 3)  # exceeds rollout length 2
                self.assertEqual(results[label]['end_date'], '2023-01-03')
                self.assertEqual(sum(results[label]['safety_mode_days'].values()), 3)
                self.assertEqual(len(results[label]['reservoirs']), 5)
                self.assertTrue(Path(results[label]['logs']).is_dir())
            self.assertTrue((Path(directory) / 'evaluation' / 'comparison.md').is_file())
            baseline_config = model / 'initial' / 'config.json'
            changed = json.loads(baseline_config.read_text())
            changed['env_args']['simulation']['end_date'] = '2023-01-02'
            baseline_config.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'configurations differ'):
                compare(model, model / 'initial', Path(directory) / 'bad_evaluation')
            self.assertFalse((Path(directory) / 'bad_evaluation').exists())


if __name__ == '__main__':
    unittest.main()
