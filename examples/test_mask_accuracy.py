"""Oracle validation, error injection, recovery coverage and training neutrality."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.mask_accuracy_oracle import (confusion, local_labels, rates,
                                            reference_graph, suffix_reachability)
from examples.train_mask_accuracy import MaskAccuracyRunner, MaskAudit, reference_masks
from examples.test_cascade_s3_recovery import fixture
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.utils.configs_tools import get_defaults_yaml_args


def fixture_case(states, stage, level_limit):
    _, env = get_defaults_yaml_args('happo', 'cascade_reservoir')
    safety = env['safety']
    safety['p3']['level_change_limit_m'] = {s.reservoir_id: level_limit for s in states}
    return dict(date='synthetic', stage=stage, dt=1., safety=safety, reservoirs=[
        dict(id=s.reservoir_id, storage=s.storage_m3, previous=s.previous_release_m3s,
             forcing=s.forcing_inflow_m3s, levels=[0., 20.], volumes=[0., 200.],
             releases=s.mapper.candidate_releases_m3s.tolist(),
             p1_low=s.p1.level.min_level_m, p1_high=s.p1.level.max_level_m,
             qmin=s.p1.release.min_release_m3s, qmax=s.p1.release.max_release_m3s,
             p2_low=s.p2.level.min_level_m, p2_high=s.p2.level.max_level_m) for s in states])


class MaskAccuracyTests(unittest.TestCase):
    def test_hand_calculated_boundaries_and_first_day(self):
        _, states = fixture()
        case = fixture_case(states, 'dry_supply', .1)
        # V=100, I=1, dt=1; Q=0/1 -> V_next=101/100. Both are on P3 bounds.
        np.testing.assert_array_equal(local_labels(case, 0, [1.], p4=None), [[1, 1]])
        np.testing.assert_array_equal(local_labels(case, 0, [1.], p4=1.), [[0, 1]])
        np.testing.assert_array_equal(local_labels(case, 0, [3.], p4=None), [[0, 0]])
        case['reservoirs'][0]['previous'] = 0.
        case['safety']['p3']['release_change_floor_m3s'] = .25
        np.testing.assert_array_equal(local_labels(case, 0, [1.], p4=None), [[1, 0]])
        # Just inside/outside physical Q cap, with only the declared numerical tolerance.
        case['reservoirs'][0]['releases'] = [1., 1.+.5e-7, 1.+2e-7]
        np.testing.assert_array_equal(local_labels(case, 0, [1.], p1_only=True), [[1, 1, 0]])

    def test_graph_against_exhaustive_suffix_paths(self):
        rng = np.random.RandomState(71)
        for _ in range(30):
            root = rng.rand(3) > .5
            edges = [rng.rand(3, 3) > .6 for _ in range(4)]
            actual_root, conditions = suffix_reachability(root, edges)
            expected_root = np.zeros(3, bool)
            for path in itertools.product(range(3), repeat=5):
                if root[path[0]] and all(edges[i][path[i], path[i+1]] for i in range(4)):
                    expected_root[path[0]] = True
            np.testing.assert_array_equal(actual_root, expected_root)
            for layer, matrix in enumerate(edges):
                expected = np.zeros((3, 3), bool)
                for suffix in itertools.product(range(3), repeat=5-layer):
                    if all(edges[layer+i][suffix[i], suffix[i+1]] for i in range(4-layer)):
                        expected[suffix[0], suffix[1]] = True
                np.testing.assert_array_equal(conditions[layer], expected)

    def test_all_five_recovery_modes_and_hard_infeasibility(self):
        scenarios = [(1., 20., 'dry_supply', 'normal'),
                     (1.1, 20., 'dry_supply', 'p4_relaxed'),
                     (2.5, 20., 'flood_control', 'p3_relaxed'),
                     (1.5, 10., 'flood_control', 'p2_relaxed'),
                     (5., 10., 'dry_supply', 'min_violation_fallback')]
        for inflow, upper, stage, mode in scenarios:
            with self.subTest(mode=mode):
                solver, states = fixture(inflow=inflow, upper=upper)
                case = fixture_case(states, stage, .1)
                result = solver.solve(states, stage)
                self.assertEqual(result.mode, mode)
                root, edges, mask, conditions, forced = reference_masks(case, result)
                np.testing.assert_array_equal(root, result.extendability.root_local_mask)
                np.testing.assert_array_equal(mask, result.extendability.root_action_mask)
                for reference, actual in zip(edges, result.extendability.compatibility_matrices):
                    np.testing.assert_array_equal(reference, actual)
                if forced is not None:
                    self.assertEqual(forced, result.forced_joint_action)
                for i, rid in enumerate(result.extendability.reservoir_order[1:]):
                    for upstream in range(edges[i].shape[0]):
                        np.testing.assert_array_equal(conditions[i][upstream],
                            result.extendability.get_action_mask(rid, upstream))
        solver, states = fixture(inflow=300.)
        root, edges = reference_graph(fixture_case(states, 'flood_control', .1), p1_only=True)
        self.assertFalse(suffix_reachability(root, edges)[0].any())
        with self.assertRaisesRegex(RuntimeError, 'No P1-safe'):
            solver.solve(states, 'flood_control')

    def test_fp_fn_injection_stops_and_saves_evidence(self):
        expected = np.array([1, 0, 1, 0], bool)
        self.assertEqual(confusion([0, 1, 1, 0], expected), dict(tp=1, tn=1, fp=1, fn=1))
        self.assertIsNone(rates(dict(tp=0, tn=4, fp=0, fn=0))['recall'])
        for predicted, field in [([1, 1, 1, 0], 'fp'), ([0, 0, 1, 0], 'fn')]:
            with tempfile.TemporaryDirectory() as tmp:
                audit = MaskAudit(Path(tmp)/'audit')
                audit.current_case = dict(date='synthetic')
                audit.current_mode = 'normal'
                try:
                    with self.assertRaisesRegex(AssertionError, 'Mask mismatch'):
                        audit.compare(predicted, expected, 'injected', 'WDD')
                    audit.write('failed')
                    evidence = json.loads((audit.directory/'first_mismatch.json').read_text())
                    self.assertEqual(evidence['counts'][field], 1)
                finally:
                    audit.close()

    def test_audit_does_not_change_real_training(self):
        with tempfile.TemporaryDirectory(prefix='mask-neutrality-') as tmp:
            algo, env = get_defaults_yaml_args('happo', 'cascade_reservoir')
            algo['train'].update(episode_length=4, num_env_steps=8, model_dir=None, save_interval=1)
            algo['eval']['use_eval'] = False
            algo['device'].update(cuda=False, torch_threads=1)
            algo['mask_accuracy'] = dict(print_every=4)
            snapshots = []
            for cls, label in [(CascadeReservoirRunner, 'plain'), (MaskAccuracyRunner, 'audit')]:
                cfg = copy.deepcopy(algo)
                cfg['logger']['log_dir'] = str(Path(tmp)/label)
                runner = cls(dict(algo='happo', env='cascade_reservoir', exp_name=label), cfg, env)
                try:
                    runner.run()
                    networks = [a.actor for a in runner.actor] + [runner.critic.critic] + [
                        c.critic for c in runner.constraint_critic.critics]
                    snapshots.append([[v.detach().clone() for v in n.state_dict().values()] for n in networks])
                    if cls is MaskAccuracyRunner:
                        summary = json.loads((runner.mask_audit.directory/'summary.json').read_text())
                        self.assertEqual(summary['status'], 'completed')
                        self.assertEqual(summary['checked_steps'], 8)
                        self.assertEqual(summary['totals']['fp'], 0)
                        self.assertEqual(summary['totals']['fn'], 0)
                finally:
                    runner.close()
            for before, after in zip(*snapshots):
                for a, b in zip(before, after):
                    self.assertTrue(torch.equal(a, b), 'Audit changed real training parameters')


if __name__ == '__main__':
    unittest.main()
