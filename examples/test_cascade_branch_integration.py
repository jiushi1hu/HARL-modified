"""Step 7: each recovery branch through real sampling, physics and training.

Run after Steps 1 and 2/3:
    python -m unittest examples.test_cascade_branch_integration -v -f
All scenario changes are in-memory test fixtures, never production parameters.
No safety result, action, transition, critic or optimizer is mocked.
"""
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from harl.envs.cascade_reservoir.cascade_reservoir_env import _INFLOW_COLUMNS
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.utils.configs_tools import get_defaults_yaml_args

ORDER = ('WDD', 'BHT', 'XLD', 'XJB', 'THR')
BATCH = 4


def configure_fixture(env, scenario):
    """Derive deterministic test inputs from real Z-V and the existing action grid."""
    p = env.physics['WDD']
    v = env.initial_storage_m3['WDD']
    h = float(p.level_from_storage(v))
    q = env.action_mappers['WDD'].candidate_releases_m3s
    inflow = env._get_current_forcing()['external_inflow_m3s']
    if scenario == 'p4_relaxed':
        index = int(np.argmin(abs(q - inflow)))
        inflow = float(q[index] + .25 * (q[index+1] - q[index]))
        next_h = float(p.level_from_storage(v + (inflow-q[index])*p.timestep_seconds))
        env.s3_solver.level_change_limit_m['WDD'] = abs(next_h-h) * 1.1
        env.daily_inflow = env.daily_inflow.copy(deep=True)
        env.daily_inflow.loc[:, _INFLOW_COLUMNS['WDD']] = inflow
    elif scenario == 'p3_relaxed':
        index = int(np.argmin(abs(q - inflow)))
        next_h = float(p.level_from_storage(v + (inflow-q[index])*p.timestep_seconds))
        assert abs(next_h-h) > 1e-6
        env.s3_solver.level_change_limit_m['WDD'] = abs(next_h-h) * .75
    elif scenario == 'p2_relaxed':
        factor = env.s3_solver.max_p3_relaxation_factor
        limit = env.s3_solver.level_change_limit_m['WDD'] * factor
        lower_storage = p.storage_from_level(max(h-limit, env.hard_min_level_m['WDD']))
        upper_storage = p.storage_from_level(min(h+limit, env.safety_upper_level_m['WDD']))
        volumes = v + (inflow-q)*p.timestep_seconds
        feasible = (volumes >= lower_storage) & (volumes <= upper_storage) & (q >= .75*inflow)
        assert feasible.any()
        lowest = min(float(p.level_from_storage(x)) for x in volumes[feasible])
        # Every P3-max candidate misses this TEST-ONLY P2 cap by at least 0.05 m.
        env.level_limits = copy.deepcopy(env.level_limits)
        env.level_limits['WDD']['normal_upper_level_m'] = lowest - .05
    elif scenario == 'min_violation_fallback':
        env.s3_solver.level_change_limit_m['WDD'] = 1e-6
    elif scenario == 'p1_infeasible':
        inflow = (env.max_storage_m3['WDD'] - v)/p.timestep_seconds + q[-1] + 1000.
        env.daily_inflow = env.daily_inflow.copy(deep=True)
        env.daily_inflow.loc[:, _INFLOW_COLUMNS['WDD']] = inflow
    env._prepared_context = None


class BranchIntegrationTests(unittest.TestCase):
    def run_scenario(self, scenario, inactive=False):
        algo, config = get_defaults_yaml_args('happo', 'cascade_reservoir')
        algo['train'].update(episode_length=BATCH, model_dir=None)
        algo['eval']['use_eval'] = False
        algo['device']['cuda'] = False
        if inactive:
            for spec in config['long_term_constraints']:
                spec['active_stages'] = ['flood_control']  # batch is January (dry_supply)
            config['lagrangian']['initial_value'] = .5  # detect an erroneous decrease
            # Test-only positive xi while inactive: it must survive storage but not enter v.
            config['long_term_constraints'][0]['requirement'] = 1e6
        args = dict(algo='happo', env='cascade_reservoir', exp_name='branch_' + scenario)
        with tempfile.TemporaryDirectory(prefix='harl-branch-') as directory:
            algo['logger']['log_dir'] = directory
            runner = CascadeReservoirRunner(args, algo, config)
            try:
                env = runner.raw_train_env
                configure_fixture(env, scenario)
                runner.warmup()
                runner.prep_rollout()
                if scenario == 'p1_infeasible':
                    before = (env.current_date, dict(env.storage_m3), dict(env.previous_release_m3s))
                    with self.assertRaisesRegex(RuntimeError, 'No P1-safe'):
                        runner.collect(0)
                    self.assertEqual(before, (env.current_date, env.storage_m3, env.previous_release_m3s))
                    self.assertEqual(runner.actor_buffer[0].step, 0)
                    return

                sampling_order = []
                for i, actor in enumerate(runner.actor):
                    original = actor.get_actions
                    def get_actions(*a, _i=i, _original=original, **kw):
                        sampling_order.append(_i)
                        return _original(*a, **kw)
                    actor.get_actions = get_actions
                modes = []
                for t in range(BATCH):
                    before = (env.current_date, dict(env.storage_m3), dict(env.previous_release_m3s))
                    context = env.prepare_step()
                    recovery = context['safety_result']
                    self.assertEqual(before, (env.current_date, env.storage_m3, env.previous_release_m3s))
                    if t == 0:
                        self.assertEqual(recovery.mode, scenario)
                        self.assert_relaxation_stage(recovery, env)
                    collected = runner.collect(t)
                    sample = runner._pending_sampling_result
                    self.assertEqual(sampling_order[-5:], list(range(5)))
                    self.assertEqual(before, (env.current_date, env.storage_m3, env.previous_release_m3s))
                    actions = tuple(int(x) for x in collected[1][0].reshape(-1))
                    if recovery.forced_joint_action is not None:
                        self.assertEqual(actions, recovery.forced_joint_action)
                    for i, key in enumerate(ORDER):
                        mask = recovery.get_action_mask(key, None if i == 0 else actions[i-1])
                        self.assertTrue(mask[actions[i]])
                        np.testing.assert_array_equal(sample.action_masks[i][0], mask)
                    # A rejected full action path must not mutate physical state.
                    if t == 0:
                        rejected = np.flatnonzero(~recovery.get_action_mask('WDD').astype(bool))
                        if rejected.size:
                            bad = np.array(actions)
                            bad[0] = rejected[0]
                            with self.assertRaises((ValueError, RuntimeError)):
                                env.step(bad)
                            self.assertEqual(before, (env.current_date, env.storage_m3, env.previous_release_m3s))
                    transition = runner.envs.step(collected[1])
                    infos = transition[4][0]
                    modes.append(infos[0]['safety_mode'])
                    for i, key in enumerate(ORDER):
                        info = infos[i]
                        q = info['executed_release_m3s']
                        self.assertEqual(q, info['target_release_m3s'])
                        self.assertEqual(q, env.action_mappers[key].action_to_release(actions[i]))
                        inflow = (context['external_inflow_m3s'] if i == 0 else
                                  infos[i-1]['executed_release_m3s'] + context['interval_inflows_m3s'][key])
                        self.assertAlmostEqual(info['inflow_m3s'], inflow)
                        self.assertAlmostEqual(info['next_storage_m3'], before[1][key] +
                                               (inflow-q)*env.physics[key].timestep_seconds, delta=.01)
                        bounds = env.p1_constraints[key]
                        self.assertGreaterEqual(info['next_level_m'], bounds.level.min_level_m-1e-8)
                        self.assertLessEqual(info['next_level_m'], bounds.level.max_level_m+1e-8)
                        if i:
                            np.testing.assert_allclose(sample.decision_observations[i][0,-1],
                                inflow/env.action_mappers[key].map_max_release_m3s, rtol=1e-6)
                        if recovery.mode != 'min_violation_fallback':
                            self.assertLessEqual(info['next_level_m'], recovery.effective_p2_bounds[i].max_level_m+1e-8)
                            self.assertGreaterEqual(info['next_level_m'], recovery.effective_p2_bounds[i].min_level_m-1e-8)
                            limit = env.s3_solver.level_change_limit_m[key] * (
                                1 + recovery.p3_relaxation_fraction*(env.s3_solver.max_p3_relaxation_factor-1))
                            old_h = env.physics[key].level_from_storage(before[1][key])
                            self.assertLessEqual(abs(info['next_level_m']-old_h), limit+1e-8)
                            previous_q = before[2][key]
                            if previous_q is not None:
                                dq = max(env.s3_solver.release_change_ratio * previous_q,
                                         env.s3_solver.release_change_floor_m3s)
                                factor = 1 + recovery.p3_relaxation_fraction * (
                                    env.s3_solver.max_p3_relaxation_factor - 1)
                                self.assertLessEqual(abs(q-previous_q), dq*factor+1e-7)
                            self.assertLessEqual(info['p2_upper_relaxation_m'],
                                                 env.s3_solver.max_upper_relaxation_m[key]+1e-8)
                            self.assertLessEqual(info['p2_lower_relaxation_m'],
                                                 env.s3_solver.max_lower_relaxation_m[key]+1e-8)
                            if recovery.p4_release_inflow_ratio is not None:
                                self.assertGreaterEqual(q, recovery.p4_release_inflow_ratio*inflow-1e-7)
                    runner.insert((*transition, *collected))
                    for i, buffer in enumerate(runner.actor_buffer):
                        np.testing.assert_array_equal(buffer.obs[t], sample.decision_observations[i])
                        np.testing.assert_array_equal(buffer.available_actions[t], sample.action_masks[i])
                        np.testing.assert_array_equal(buffer.actions[t], sample.actions[:,i])
                        np.testing.assert_array_equal(buffer.action_log_probs[t], sample.action_log_probs[:,i])
                    np.testing.assert_array_equal(runner.critic_buffer.share_obs[t], context['share_observation'])
                    cb = runner.constraint_buffer
                    np.testing.assert_array_equal(cb.raw_violations[t,0], infos[0]['constraint_raw_violations'])
                    np.testing.assert_array_equal(cb.active_flags[t,0], infos[0]['constraint_active_flags'])
                    for j in range(cb.num_constraints):
                        np.testing.assert_array_equal(cb.get_buffer(j).share_obs[t], context['share_observation'])
                runner.compute()
                np.testing.assert_allclose(cb.costs, cb.active_flags * np.maximum(0.,
                    cb.raw_violations / cb.normalizers - cb.tolerances))
                self.assertTrue(np.isfinite(cb.returns).all())
                self.assertTrue(np.isfinite(runner.critic_buffer.returns).all())
                snapshot = runner.lagrangian_manager.snapshot()
                stats = cb.compute_discounted_active_statistics()
                if inactive:
                    self.assertTrue(np.all(cb.raw_violations[:, 0, 0] > 0.0))
                    self.assertFalse(cb.active_flags.any())
                    self.assertFalse(cb.costs.any())
                    self.assertFalse(stats.valid_flags.any())
                    self.assertTrue(np.isnan(stats.discounted_active_means).all())
                old_values = runner.critic_buffer.value_preds[:-1]
                if runner.value_normalizer is not None:
                    old_values = runner.value_normalizer.denormalize(old_values)
                expected_advantage = runner.lagrangian_manager.combine_advantages(
                    runner.critic_buffer.returns[:-1] - old_values,
                    runner.constraint_critic.compute_advantages(cb), snapshot,
                )
                expected_factor = np.ones_like(expected_advantage)
                final_ratios = []
                events = []
                for i, critic in enumerate([runner.critic] + list(runner.constraint_critic.critics)):
                    original = critic.train
                    def train_critic(*a, _i=i, _original=original, **kw):
                        events.append(('critic', _i))
                        np.testing.assert_array_equal(runner.lagrangian_manager.snapshot(), snapshot)
                        return _original(*a, **kw)
                    critic.train = train_critic
                for i, actor in enumerate(runner.actor):
                    original = actor.train
                    def train_actor(*a, _i=i, _original=original, **kw):
                        events.append(('actor', _i))
                        np.testing.assert_array_equal(runner.lagrangian_manager.snapshot(), snapshot)
                        buffer, advantage = a[:2]
                        np.testing.assert_allclose(advantage, expected_advantage)
                        np.testing.assert_allclose(buffer.factor, expected_factor, rtol=1e-5, atol=1e-7)
                        result = _original(*a, **kw)
                        with torch.no_grad():
                            logp, _, _ = runner.actor[_i].evaluate_actions(
                                buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]),
                                buffer.rnn_states[0:1].reshape(-1, *buffer.rnn_states.shape[2:]),
                                buffer.actions.reshape(-1, 1), buffer.masks[:-1].reshape(-1, 1),
                                buffer.available_actions[:-1].reshape(-1, buffer.available_actions.shape[-1]),
                                buffer.active_masks[:-1].reshape(-1, 1),
                            )
                        ratio = np.exp(logp.cpu().numpy() - buffer.action_log_probs.reshape(-1, 1))
                        self.assertTrue(np.isfinite(ratio).all())
                        expected_factor[:] *= ratio.reshape(expected_factor.shape)
                        final_ratios.append(ratio)
                        return result
                    actor.train = train_actor
                original = runner.lagrangian_manager.update
                def update_lambda(*a, **kw):
                    events.append(('lambda', 0))
                    return original(*a, **kw)
                runner.lagrangian_manager.update = update_lambda
                runner.prep_training()
                order = [4,2,0,3,1]
                original_randperm = torch.randperm
                def training_permutation(n, *a, **kw):
                    # torch is shared across modules: do not intercept minibatch shuffles.
                    return torch.tensor(order) if n == 5 else original_randperm(n, *a, **kw)
                with patch('harl.runners.cascade_reservoir_runner.torch.randperm', side_effect=training_permutation):
                    actor_info, critic_info = runner.train()
                self.assertEqual(events, [('critic',i) for i in range(1+cb.num_constraints)] +
                                 [('actor',i) for i in order] + [('lambda',0)])
                for info in actor_info + [critic_info]:
                    self.assertTrue(all(np.isfinite(float(value)) for value in info.values()))
                if scenario == 'min_violation_fallback':
                    # Each mask permits exactly one action: its probability is 1, not a policy failure.
                    for ratio in final_ratios:
                        np.testing.assert_allclose(ratio, 1., atol=1e-7)
                elif scenario == 'normal':
                    self.assertTrue(any(np.any(abs(ratio-1.) > 1e-6) for ratio in final_ratios))
                if inactive:
                    np.testing.assert_array_equal(runner.lagrangian_manager.snapshot(), snapshot)
                    self.assertTrue(np.all(snapshot > 0))
                runner.after_update()
                self.assertFalse(cb.raw_violations.any())
                self.assertFalse(cb.active_flags.any())
                self.assertFalse(cb.costs.any())
                print('PASS branch:', scenario, 'inactive:', inactive, 'rollout modes:', modes, flush=True)
            finally:
                runner.close()

    def assert_relaxation_stage(self, result, env):
        self.assertGreater(np.count_nonzero(result.get_action_mask('WDD')), 0)
        mode = result.mode
        if mode == 'normal':
            self.assertEqual(result.p3_relaxation_fraction, 0.)
            self.assertEqual(result.p2_relaxation_fraction, 0.)
        elif mode == 'p4_relaxed':
            self.assertGreaterEqual(result.p4_release_inflow_ratio, env.s3_solver.min_p4_ratio)
            self.assertLess(result.p4_release_inflow_ratio, env.s3_solver.normal_p4_ratio)
            self.assertEqual(result.p3_relaxation_fraction, 0.)
            self.assertEqual(result.p2_relaxation_fraction, 0.)
        elif mode == 'p3_relaxed':
            self.assertEqual(result.p4_release_inflow_ratio, env.s3_solver.min_p4_ratio)
            self.assertGreater(result.p3_relaxation_fraction, 0.)
            self.assertLessEqual(result.p3_relaxation_fraction, 1.)
            self.assertEqual(result.p2_relaxation_fraction, 0.)
        else:
            self.assertEqual(result.p3_relaxation_fraction, 1.)
            self.assertGreater(result.p2_relaxation_fraction, 0.)
            self.assertLessEqual(result.p2_relaxation_fraction, 1.)
        self.assertEqual(result.forced_joint_action is not None, mode == 'min_violation_fallback')

    def test_00_normal(self): self.run_scenario('normal')
    def test_01_p4(self): self.run_scenario('p4_relaxed')
    def test_02_p3(self): self.run_scenario('p3_relaxed')
    def test_03_p2(self): self.run_scenario('p2_relaxed')
    def test_04_fallback(self): self.run_scenario('min_violation_fallback')
    def test_05_inactive(self): self.run_scenario('normal', inactive=True)
    def test_06_p1_infeasible(self): self.run_scenario('p1_infeasible')


if __name__ == '__main__':
    unittest.main()
