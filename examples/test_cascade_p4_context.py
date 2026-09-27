"""P4 context/S201 contract; fixture parameters are synthetic test data only."""
import unittest
from dataclasses import replace

import numpy as np

from examples.test_cascade_s3_recovery import fixture
from harl.envs.cascade_reservoir.safety.constraint_context import (
    ConstraintContext, P4Constraints, resolve_s201_bounds,
)
from harl.envs.cascade_reservoir.safety.s201_local_feasibility import evaluate_local_feasibility


class P4ContextTests(unittest.TestCase):
    def evaluate(self, state, inflow, context):
        return evaluate_local_feasibility(
            physics=state.physics, mapper=state.mapper, storage_m3=state.storage_m3,
            inflow_m3s=inflow, constraints=context,
        )

    def test_s201_itself_applies_p4_and_reports_bounds(self):
        _, states = fixture()
        state = states[0]
        inactive = ConstraintContext(state.p1)
        active = replace(inactive, p4=P4Constraints(1.0))
        np.testing.assert_array_equal(self.evaluate(state, 0.5, inactive).feasible_mask, [True, True])
        result = self.evaluate(state, 0.5, active)
        np.testing.assert_array_equal(result.feasible_mask, [False, True])
        self.assertEqual(result.bounds.min_release_m3s, 0.5)
        self.assertEqual(result.safe_min_release_m3s, 0.5)
        self.assertIs(active.p1, state.p1)
        self.assertEqual(active.p4.release_inflow_ratio, 1.0)

    def test_each_inflow_resolves_again_and_zero_is_valid(self):
        _, states = fixture()
        state = states[0]
        context = ConstraintContext(state.p1, p4=P4Constraints(1.0))
        for inflow, expected in [(0.0, [True, True]), (0.5, [False, True]), (1.5, [False, False])]:
            result = self.evaluate(state, inflow, context)
            np.testing.assert_array_equal(result.feasible_mask, expected)
            self.assertEqual(result.bounds.min_release_m3s, inflow)
        self.assertEqual(context.p4.release_inflow_ratio, 1.0)

    def test_conflict_with_p1_is_empty_without_interpolation_or_relaxation(self):
        _, states = fixture()
        state = states[0]
        result = self.evaluate(state, 2.0, ConstraintContext(state.p1, p4=P4Constraints(1.0)))
        self.assertTrue(result.bounds.release_is_empty)
        self.assertFalse(result.feasible_mask.any())
        self.assertIsNone(result.safe_min_release_m3s)
        self.assertEqual(state.p1.release.max_release_m3s, 1.0)

    def test_context_requires_inflow_and_validates_p4(self):
        _, states = fixture()
        context = ConstraintContext(states[0].p1, p4=P4Constraints(0.75))
        with self.assertRaisesRegex(ValueError, 'inflow'):
            context.resolve()
        for value in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                P4Constraints(value)
            with self.assertRaises(ValueError):
                context.resolve(inflow_m3s=value)
        with self.assertRaises(TypeError):
            replace(context, p4=0.75)
        with self.assertRaises(ValueError):
            resolve_s201_bounds(context, p4=P4Constraints(1.0), inflow_m3s=1.0)
        separate = resolve_s201_bounds(states[0].p1, p4=P4Constraints(0.75), inflow_m3s=1.0)
        self.assertEqual(separate, context.resolve(inflow_m3s=1.0))

    def test_s202_rows_match_candidate_inflow_and_previous_mask_rule(self):
        solver, states = fixture(inflow=1.0)
        states = tuple(replace(state, forcing_inflow_m3s=0.1) if i else state
                       for i, state in enumerate(states))
        for ratio in (None, 1.0, 0.75):
            graph = solver._evaluate_s2(states=states, use_p2=True, p3_factor=1.0, p4_ratio=ratio)
            for i, compatibility in enumerate(graph.compatibility_results):
                state = states[i+1]
                context = solver._build_context(state=state, use_p2=True, p3_factor=1.0, p4_ratio=None)
                for upstream, inflow in enumerate(compatibility.candidate_inflows_m3s):
                    self.assertAlmostEqual(inflow, states[i].mapper.candidate_releases_m3s[upstream] + 0.1)
                    expected = self.evaluate(state, inflow, context).feasible_mask.copy()
                    if ratio is not None:
                        expected &= state.mapper.candidate_releases_m3s >= ratio * inflow - 1e-7
                    np.testing.assert_array_equal(compatibility.compatibility_matrix[upstream], expected)
            if ratio == 1.0:
                # Root is locally feasible, but no full path survives downstream P4.
                self.assertTrue(graph.wdd_local_result.feasible_mask.any())
                self.assertFalse(graph.extendability.has_joint_feasible_action)

    def test_s3_passes_ratio_or_inactive_context_without_hidden_mask(self):
        solver, states = fixture()
        for ratio in (None, 1.0, 0.75):
            context = solver._build_context(state=states[0], use_p2=True, p3_factor=1.0, p4_ratio=ratio)
            if ratio is None:
                self.assertIsNone(context.p4)
            else:
                self.assertEqual(context.p4.release_inflow_ratio, ratio)
            direct = self.evaluate(states[0], 0.5, context)
            wrapped = solver._evaluate_local(state=states[0], inflow_m3s=0.5, use_p2=True,
                                             p3_factor=1.0, p4_ratio=ratio)
            self.assertEqual(direct.bounds, wrapped.bounds)
            np.testing.assert_array_equal(direct.feasible_mask, wrapped.feasible_mask)


if __name__ == '__main__':
    unittest.main()
