"""Synthetic xi/chi/cost trajectory tests; production parameters are unchanged."""
import copy
import unittest
import numpy as np
from harl.common.buffers.constraint_buffer import ConstraintBuffer
from harl.envs.cascade_reservoir.constraint_costs import ConstraintSpec, ConstraintCostEvaluator
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


def make_buffer():
    args = dict(episode_length=3, n_rollout_threads=1, hidden_sizes=[2], recurrent_n=1,
                gamma=.5, gae_lambda=1., use_gae=True, use_proper_time_limits=False)
    return ConstraintBuffer(args, [2], ['a', 'b', 'c'], [0., 0., 0.], [2., 1., .5], [.1, .2, 0.])


class ConstraintPipelineTests(unittest.TestCase):
    def test_raw_formula_all_types_and_inactive_is_not_erased(self):
        for kind in ('eco_release', 'navigation_release', 'guaranteed_power'):
            evaluator = ConstraintCostEvaluator([ConstraintSpec('WDD', kind, 100.,
                normalizer=2., tolerance=.1, active_stages=('dry_supply',))])
            for stage, flag in [('dry_supply', 1), ('flood_control', 0)]:
                result = evaluator.evaluate({'WDD': 75.}, {'WDD': 75.}, stage)
                np.testing.assert_allclose(result.raw_violations, [.5])
                np.testing.assert_allclose(result.normalized_violations, [.25])
                np.testing.assert_allclose(result.costs, [.15 * flag])
                np.testing.assert_array_equal(result.active_flags, [flag])
        with self.assertRaisesRegex(ValueError, 'beta'):
            ConstraintSpec('WDD', 'eco_release', 100., scale_coefficient=2.)

    def test_buffer_time_indices_rebuild_gae_statistics_and_reset(self):
        buffer = make_buffer()
        raw = np.array([[.8, .4, .9], [.4, .7, .8], [.2, .6, .7]], dtype=np.float32)
        flags = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        expected = flags * np.maximum(0., raw / [2., 1., .5] - [.1, .2, 0.])
        buffer.initialize(np.array([[10., 11.]], dtype=np.float32))
        for t in range(3):
            source = raw[t:t+1].copy()
            buffer.insert(share_obs=np.array([[t+20., t+21.]]),
                rnn_states_critic=np.zeros((1,3,1,2)), value_preds=np.zeros((1,3)),
                costs=expected[t:t+1], raw_violations=source, active_flags=flags[t:t+1],
                masks=np.ones((1,1)), bad_masks=np.ones((1,1)))
            source[:] = 999.
            np.testing.assert_array_equal(buffer.raw_violations[t, 0], raw[t])
            np.testing.assert_array_equal(buffer.active_flags[t, 0], flags[t])
        # Buffer wrap-around must not erase a completed trajectory before compute.
        self.assertEqual(buffer.step, 0)
        for j in range(3):
            buffer.get_buffer(j).rewards[:] = 999.  # compute must rebuild from saved xi/chi
        buffer.compute_returns(np.zeros((1,3)))
        np.testing.assert_allclose(buffer.costs[:,0], expected, atol=1e-7)
        returns = expected.copy()
        for t in (1, 0):
            returns[t] += .5 * returns[t+1]
        np.testing.assert_allclose(buffer.returns[:-1,0], returns, atol=1e-7)
        np.testing.assert_allclose(buffer.compute_advantages()[:,0], returns, atol=1e-7)
        stats = buffer.compute_discounted_active_statistics()
        np.testing.assert_allclose(stats.discounted_active_means[:2], [(0.3+.5*.1)/1.5, (.5*.5+.25*.4)/.75])
        self.assertTrue(np.isnan(stats.discounted_active_means[2]))
        np.testing.assert_array_equal(stats.valid_flags, [True, True, False])
        np.testing.assert_array_equal(stats.active_counts, [2,2,0])
        buffer.after_update()
        self.assertFalse(buffer.raw_violations.any())
        self.assertFalse(buffer.active_flags.any())
        self.assertFalse(buffer.costs.any())

    def test_runner_checks_raw_ids_shapes_and_agent_consistency(self):
        runner = CascadeReservoirRunner.__new__(CascadeReservoirRunner)
        runner.constraint_buffer = make_buffer()
        runner.num_agents, runner.num_constraints = 5, 3
        runner.constraint_ids = ('a','b','c')
        info = dict(constraint_ids=runner.constraint_ids, constraint_raw_violations=np.array([.8,.4,.9]),
                    constraint_active_flags=np.array([1.,0.,0.]), constraint_costs=np.array([.3,0.,0.]))
        infos = [[copy.deepcopy(info) for _ in range(5)]]
        costs, flags, raw = runner._extract_constraint_feedback(infos)
        np.testing.assert_allclose(raw[0], info['constraint_raw_violations'])
        for field, value in [('constraint_raw_violations', [.8,.4]),
                             ('constraint_raw_violations', [.8,.4,float('nan')]),
                             ('constraint_raw_violations', [-1.,.4,.9]),
                             ('constraint_active_flags', [.5,0,0]),
                             ('constraint_costs', [.3,1,0]),
                             ('constraint_costs', [.3,1e-9,0]),
                             ('constraint_ids', ('b','a','c'))]:
            bad = copy.deepcopy(infos)
            bad[0][0][field] = value
            with self.assertRaises(ValueError):
                runner._extract_constraint_feedback(bad)
        bad = copy.deepcopy(infos)
        bad[0][1]['constraint_raw_violations'][2] = .1  # cost remains zero but xi must agree
        with self.assertRaisesRegex(ValueError, 'differs'):
            runner._extract_constraint_feedback(bad)
        del bad[0][0]['constraint_raw_violations']
        with self.assertRaises(KeyError):
            runner._extract_constraint_feedback(bad)

    def test_invalid_transform_parameters(self):
        buffer = make_buffer()
        with self.assertRaises(ValueError):
            buffer.prepare_training_costs([[-.1, 0, 0]], [[1,1,1]])
        with self.assertRaises(ValueError):
            buffer.prepare_training_costs([[.1, 0, 0]], [[1,.5,1]])


if __name__ == '__main__':
    unittest.main()
