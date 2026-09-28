import unittest

import numpy as np

from harl.runners.cascade_reservoir_runner import summarize_lagrangian_advantages


class AdvantageDiagnosticsTests(unittest.TestCase):
    def test_signed_penalties_cancel_and_inputs_are_unchanged(self):
        reward = np.array([[[2.]], [[-2.]]])
        costs = np.array([[[4., -2.]], [[-4., 2.]]])
        weights = np.array([.5, 1.])
        originals = [x.copy() for x in (reward, costs, weights)]
        info = summarize_lagrangian_advantages(reward, costs, weights, ('a', 'b'))
        self.assertEqual(info['advantage/reward_rms'], 2.)
        self.assertEqual(info['advantage/weighted_constraint_rms/a'], 2.)
        self.assertEqual(info['advantage/penalty_rms'], 0.)
        self.assertEqual(info['advantage/sign_flip_fraction'], 0.)
        for x, original in zip((reward, costs, weights), originals):
            np.testing.assert_array_equal(x, original)

    def test_sign_reversal_and_zero_reward(self):
        info = summarize_lagrangian_advantages(np.ones((2, 1, 1)),
                    np.array([[[2.]], [[0.]]]), np.array([1.]), ('a',))
        self.assertEqual(info['advantage/sign_flip_fraction'], .5)
        info = summarize_lagrangian_advantages(np.zeros((2, 1, 1)),
                    np.ones((2, 1, 1)), np.array([1.]), ('a',))
        self.assertNotIn('advantage/penalty_to_reward_rms', info)


if __name__ == '__main__':
    unittest.main()
