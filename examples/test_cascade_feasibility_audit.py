"""Independent exhaustive checks for the diagnostic graph traversal."""
import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from examples.diagnose_cascade_feasibility import executable_nodes, path_nodes


class FeasibilityAuditTests(unittest.TestCase):
    def test_matches_exhaustive_complete_paths(self):
        rng = np.random.RandomState(23)
        for _ in range(80):
            root = rng.rand(3) > .3
            matrices = [rng.rand(3, 3) > .6 for _ in range(4)]
            allowed = [rng.rand(3) > .25 for _ in range(5)]
            expected = [np.zeros(3, dtype=bool) for _ in range(5)]
            for actions in itertools.product(range(3), repeat=5):
                if (root[actions[0]] and all(allowed[i][a] for i, a in enumerate(actions))
                        and all(matrices[i][actions[i], actions[i+1]] for i in range(4))):
                    for i, a in enumerate(actions):
                        expected[i][a] = True
            for actual, wanted in zip(path_nodes(root, matrices, allowed), expected):
                np.testing.assert_array_equal(actual, wanted)

    def test_individual_feasibility_does_not_imply_joint_feasibility(self):
        root = np.ones(2, dtype=bool)
        matrices = [np.eye(2, dtype=bool) for _ in range(4)]
        allowed = [np.ones(2, dtype=bool) for _ in range(5)]
        allowed[0] = np.array([True, False])
        allowed[-1] = np.array([False, True])
        self.assertFalse(any(x.any() for x in path_nodes(root, matrices, allowed)))

    def test_fallback_respects_forced_path(self):
        safety = SimpleNamespace(
            forced_joint_action=(0, 0, 0, 0, 0),
            extendability=SimpleNamespace(root_local_mask=np.ones(2, dtype=bool),
                                          compatibility_matrices=[np.ones((2, 2), dtype=bool)]*4))
        allowed = [np.ones(2, dtype=bool) for _ in range(5)]
        for mask in executable_nodes(safety, allowed):
            np.testing.assert_array_equal(mask, [True, False])
        allowed[-1] = np.array([False, True])
        self.assertFalse(executable_nodes(safety, allowed)[0].any())


if __name__ == '__main__':
    unittest.main()
