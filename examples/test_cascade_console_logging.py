"""Console diagnostics must distinguish inactive constraints and cost definitions."""
import io
import unittest
from contextlib import redirect_stdout

from harl.envs.cascade_reservoir.cascade_reservoir_logger import CascadeReservoirLogger


class ConsoleLoggingTests(unittest.TestCase):
    def test_active_and_inactive_metrics(self):
        logger = CascadeReservoirLogger.__new__(CascadeReservoirLogger)
        logger.constraint_ids = ('WDD:eco_release', 'THR:eco_release')
        logger.reservoir_order = ('WDD',)
        logger._rollout_metrics = {
            'system/reward': [1., 1., 1.],
            'constraint/WDD:eco_release/cost': [0., .4],
            'constraint/WDD:eco_release/xi': [.1, .5],
        }
        info = {'value_loss': .25}
        for cid in logger.constraint_ids:
            info.update({f'lambda_before/{cid}': .5, f'lambda/{cid}': .5,
                         f'budget/{cid}': .02, f'constraint_critic/{cid}/value_loss': .1})
        info.update({'mean_cost/WDD:eco_release': .1, 'budget_gap/WDD:eco_release': .08})
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger._print_training_details([dict(policy_loss=.2, dist_entropy=.3, ratio=1.1)], info)
        text = stream.getvalue()
        active = next(line for line in text.splitlines() if 'WDD:eco_release:' in line)
        inactive = next(line for line in text.splitlines() if 'THR:eco_release:' in line)
        for field in ['cost_discounted=0.1', 'cost_mean=0.2', 'xi_mean=0.3',
                      'active=2/3', 'violation_samples=1', 'lambda=0.5 -> 0.5']:
            self.assertIn(field, active)
        for field in ['cost_discounted=N/A', 'cost_mean=N/A', 'active=0/3',
                      'gap=N/A', 'inactive: lambda unchanged']:
            self.assertIn(field, inactive)
        self.assertIn('mean_ratio=1.1', text)
        self.assertIn('Reward Critic: value_loss=0.25', text)


if __name__ == '__main__':
    unittest.main()
