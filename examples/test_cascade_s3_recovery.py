"""S3 branch and complete-path tests using explicitly synthetic linear reservoirs.

Run: python -m unittest examples.test_cascade_s3_recovery -v
Synthetic numbers below are fixtures, never production physical parameters.
"""
import csv
import itertools
import tempfile
from pathlib import Path
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from harl.envs.cascade_reservoir.action_mapping import ReleaseActionMapper
from harl.envs.cascade_reservoir.s3_relaxation import ReservoirSafetyState, S3RelaxationSolver
from harl.envs.cascade_reservoir.safety.constraint_context import LevelBounds, P1Constraints, P2Constraints, ReleaseBounds
from harl.utils.configs_tools import get_defaults_yaml_args

ORDER = ('WDD', 'BHT', 'XLD', 'XJB', 'THR')


class LinearPhysics:
    timestep_seconds = 1.0

    @staticmethod
    def storage_from_level(level):
        return np.asarray(level) * 10.0

    @staticmethod
    def level_from_storage(storage):
        return np.asarray(storage) / 10.0


def fixture(inflow=1.0, upper=20.0, level_limit=0.1):
    _, config = get_defaults_yaml_args('happo', 'cascade_reservoir')
    safety = config['safety']
    safety['p3']['level_change_limit_m'] = {key: level_limit for key in ORDER}
    solver = S3RelaxationSolver(ORDER, safety['p3'], safety['p4'], safety['relaxation'],
                                safety['p2'], safety['fallback'])
    states = tuple(ReservoirSafetyState(
        key, LinearPhysics(), ReleaseActionMapper(key, 2, 0.0, 1.0), 100.0,
        inflow if i == 0 else 0.0, None,
        P1Constraints(LevelBounds(0.0, 20.0), ReleaseBounds(0.0, 1.0)),
        P2Constraints(LevelBounds(0.0, upper if i == 0 else 20.0)),
    ) for i, key in enumerate(ORDER))
    return solver, states


def sample_first_path(result):
    path = []
    for key in ORDER:
        mask = result.get_action_mask(key, None if not path else path[-1])
        path.append(int(np.flatnonzero(mask)[0]))
    return tuple(path)


class RecoveryTests(unittest.TestCase):
    def test_normal_stops_without_relaxation(self):
        solver, states = fixture()
        with patch.object(solver, '_evaluate_s2', wraps=solver._evaluate_s2) as evaluate:
            result = solver.solve(states, 'dry_supply')
        self.assertEqual(result.mode, 'normal')
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(sample_first_path(result), (1, 1, 1, 1, 1))

    def test_p4_minimum_then_stop(self):
        solver, states = fixture(inflow=1.1)
        result = solver.solve(states, 'dry_supply')
        self.assertEqual(result.mode, 'p4_relaxed')
        self.assertAlmostEqual(result.p4_release_inflow_ratio, 1.0 / 1.1, delta=0.001)
        self.assertEqual(result.p3_relaxation_fraction, 0.0)
        self.assertEqual(result.p2_relaxation_fraction, 0.0)
        self.assertIsNone(result.forced_joint_action)
        self.assertEqual(sample_first_path(result), (1, 1, 1, 1, 1))

    def test_p3_minimum_and_inactive_p4(self):
        solver, states = fixture(inflow=2.5)
        result = solver.solve(states, 'flood_control')
        self.assertEqual(result.mode, 'p3_relaxed')
        self.assertAlmostEqual(result.p3_relaxation_fraction, 0.5, delta=0.001)
        self.assertIsNone(result.p4_release_inflow_ratio)
        self.assertEqual(result.p2_relaxation_fraction, 0.0)

    def test_p2_bounded_minimum_full_s2_and_no_mutation(self):
        solver, states = fixture(inflow=1.5, upper=10.0)
        with patch.object(solver, '_evaluate_s2', wraps=solver._evaluate_s2) as evaluate:
            result = solver.solve(states, 'flood_control')
        self.assertEqual(result.mode, 'p2_relaxed')
        self.assertAlmostEqual(result.p2_relaxation_fraction, 0.25, delta=0.001)
        self.assertIsNone(result.forced_joint_action)
        self.assertEqual(states[0].p2.level.max_level_m, 10.0)
        self.assertEqual(states[0].storage_m3, 100.0)
        self.assertLessEqual(result.effective_p2_bounds[0].max_level_m, 10.2)
        self.assertGreaterEqual(result.effective_p2_bounds[0].max_level_m, 10.05 - 1e-8)
        for call in evaluate.call_args_list:
            candidate = call.kwargs['states'][0]
            self.assertIs(candidate.p1, states[0].p1)
            self.assertLessEqual(candidate.p2.level.max_level_m, 10.2)
        self.assertEqual(sample_first_path(result)[0], 1)

    def test_p1_limits_cap_and_lower_unchanged(self):
        solver, states = fixture(upper=19.95)
        relaxed = solver._relax_p2_states(states, 1.0)
        self.assertEqual(relaxed[0].p2.level.max_level_m, 20.0)
        self.assertEqual(relaxed[0].p2.level.min_level_m, 0.0)
        self.assertIs(relaxed[0].p1, states[0].p1)

    def test_all_maxima_before_final_and_bruteforce_complete_paths(self):
        solver, states = fixture(inflow=5.0, upper=10.0)
        with patch.object(solver, '_evaluate_s2', wraps=solver._evaluate_s2) as evaluate:
            result = solver.solve(states, 'dry_supply')
        self.assertEqual(result.mode, 'min_violation_fallback')
        calls = [call.kwargs for call in evaluate.call_args_list]
        self.assertEqual([call['p4_ratio'] for call in calls], [1.0, 0.75, 0.75, 0.75, None])
        self.assertEqual([call['p3_factor'] for call in calls], [1.0, 1.0, 2.0, 2.0, None])
        self.assertAlmostEqual(calls[-2]['states'][0].p2.level.max_level_m, 10.2)
        self.assertFalse(calls[-1]['use_p2'])
        candidates = []
        for path in itertools.product(range(2), repeat=5):
            total = np.zeros(3)
            for i, (state, action) in enumerate(zip(states, path)):
                inflow = state.forcing_inflow_m3s + (path[i-1] if i else 0.0)
                level = 10.0 + (inflow - action) / 10.0
                _, scores = solver.operational_violation_components(state, level, action, inflow, True)
                total += scores
            candidates.append((tuple(total), path))
        expected_cost, expected_path = min(candidates)
        self.assertEqual(result.forced_joint_action, expected_path)
        self.assertEqual(sample_first_path(result), expected_path)
        np.testing.assert_allclose([result.p2_total_violation, result.p3_total_violation,
                                   result.p4_total_violation], expected_cost)

    def test_p1_infeasible_is_explicit_error(self):
        solver, states = fixture(inflow=300.0)
        with self.assertRaisesRegex(RuntimeError, 'No P1-safe'):
            solver.solve(states, 'flood_control')

    def test_raw_units_normalization_first_day_and_inactive_p4(self):
        solver, states = fixture(upper=10.0)
        raw, score = solver.operational_violation_components(states[0], 10.4, 0.0, 5.0, True)
        np.testing.assert_allclose(raw, [0.0, 0.4, 0.3, 0.0, 5.0])
        np.testing.assert_allclose(score, [0.02, 3.0, 1.0])
        # History activates only the P3 release term; positive configured scale.
        solver.release_change_floor_m3s = 0.2
        solver.release_change_ratio = 0.0
        state = replace(states[0], previous_release_m3s=1.0)
        raw, score = solver.operational_violation_components(state, 10.4, 0.0, 5.0, False)
        np.testing.assert_allclose(raw, [0.0, 0.4, 0.3, 0.8, 0.0])
        np.testing.assert_allclose(score, [0.02, 7.0, 0.0])
        _, score = solver.operational_violation_components(states[0], 10.0, 0.0, 0.0, True)
        self.assertTrue(np.isfinite(score).all())

    def test_dp_matches_exhaustive_graph_including_unreachable_edges_and_ties(self):
        rng = np.random.RandomState(41)
        for _ in range(40):
            root = rng.randint(0, 3, size=(3, 3)).astype(float)
            edges = [rng.randint(0, 3, size=(3, 3, 3)).astype(float) for _ in range(4)]
            for edge in edges:
                edge[rng.rand(3, 3) < 0.15] = np.inf
                edge[0, 0] = 0.0  # ensure a complete path
            expected = []
            for path in itertools.product(range(3), repeat=5):
                cost = root[path[0]].copy()
                for i, edge in enumerate(edges):
                    cost += edge[path[i], path[i+1]]
                if np.isfinite(cost).all():
                    expected.append((tuple(cost), path))
            path, cost = S3RelaxationSolver._minimum_lexicographic_path(root, edges)
            self.assertEqual((tuple(cost), path), min(expected))
        root = np.zeros((2, 3))
        edges = [np.zeros((2, 2, 3)) for _ in range(4)]
        self.assertEqual(S3RelaxationSolver._minimum_lexicographic_path(root, edges)[0], (0,)*5)

    def test_higher_priority_cannot_be_offset_by_lower_priority(self):
        root = np.array([[0.0, 10.0, 10.0], [0.01, 0.0, 0.0]])
        self.assertEqual(S3RelaxationSolver._minimum_lexicographic_path(root, [])[0], (0,))
        root = np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 100.0]])
        self.assertEqual(S3RelaxationSolver._minimum_lexicographic_path(root, [])[0], (1,))

    def test_real_environment_fallback_diagnostics_and_csv(self):
        from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
        from harl.envs.cascade_reservoir.cascade_reservoir_logger import CascadeReservoirLogger
        algo, config = get_defaults_yaml_args('happo', 'cascade_reservoir')
        # Test-only narrow P3 makes the actual discrete-action environment exercise final fallback.
        config['safety']['p3']['level_change_limit_m']['WDD'] = 1e-6
        env = CascadeReservoirEnv(config)
        try:
            env.reset()
            recovery = env.prepare_step()['safety_result']
            self.assertEqual(recovery.mode, 'min_violation_fallback')
            transition = env.step(np.asarray(sample_first_path(recovery)))
            infos = transition[4]
            np.testing.assert_allclose(
                [infos[0][key] for key in ('p2_total_violation', 'p3_total_violation', 'p4_total_violation')],
                [recovery.p2_total_violation, recovery.p3_total_violation, recovery.p4_total_violation],
                rtol=1e-7, atol=1e-8,
            )
            with tempfile.TemporaryDirectory(prefix='harl-s3-log-test-') as directory:
                logger = CascadeReservoirLogger({}, algo, config, 5, Mock(), directory)
                try:
                    logger.init(1)
                    logger.episode_init(1)
                    logger._record_environment_step('train', np.asarray([transition[2]]),
                                                    np.asarray([transition[3]]), [infos])
                finally:
                    logger.close()
                with (Path(directory) / 'cascade_system_timeseries.csv').open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row['safety_mode'], 'min_violation_fallback')
                self.assertEqual(float(row['p2_relaxation_fraction']), 1.0)
                self.assertGreater(float(row['p3_total_violation']), 0.0)
                with (Path(directory) / 'cascade_reservoir_timeseries.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 5)
                for info, row in zip(infos, rows):
                    for key in ('p2_recovery_upper_level_m', 'p2_upper_relaxation_m',
                                'p2_upper_violation_m', 'p3_level_violation_m', 'p4_release_violation_m3s'):
                        self.assertEqual(float(row[key]), info[key])
                    self.assertEqual(info['executed_release_m3s'], info['target_release_m3s'])
        finally:
            env.close()

    def test_missing_or_invalid_configuration_fails(self):
        _, config = get_defaults_yaml_args('happo', 'cascade_reservoir')
        safety = config['safety']
        safety['p2']['max_upper_relaxation_m']['WDD'] = -0.2
        with self.assertRaises(ValueError):
            S3RelaxationSolver(ORDER, safety['p3'], safety['p4'], safety['relaxation'],
                              safety['p2'], safety['fallback'])
        safety['p2']['max_upper_relaxation_m']['WDD'] = 0.2
        with self.assertRaises(ValueError):
            S3RelaxationSolver(ORDER, safety['p3'], safety['p4'], safety['relaxation'], safety['p2'], {})


if __name__ == '__main__':
    unittest.main()
