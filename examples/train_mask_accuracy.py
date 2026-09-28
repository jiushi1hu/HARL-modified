"""Train HAPPO while independently auditing every generated action mask."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from examples.mask_accuracy_oracle import (
    MODES, confusion, effective_parameters, fallback_path, make_case,
    rates, reference_graph, suffix_reachability,
)
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


def reference_masks(case, safety):
    parameters = effective_parameters(case, safety)
    root, edges = reference_graph(case, **parameters)
    root_mask, conditional = suffix_reachability(root, edges)
    if not root_mask.any():
        raise AssertionError('Reported recovery has no independent complete path')
    # Verify recovery was necessary; do not legitimize an arbitrary relaxation
    # merely because the solver reports it as the effective context.
    cfg = case['safety']
    p4 = cfg['p4']['normal_release_inflow_ratio'] if case['stage'] == 'dry_supply' else None
    prior = []
    if safety.mode != 'normal':
        prior.append(dict(p4=p4))
    if safety.mode in ('p3_relaxed', 'p2_relaxed', 'min_violation_fallback'):
        p4 = cfg['p4']['min_release_inflow_ratio'] if p4 is not None else None
        prior.append(dict(p4=p4))
    if safety.mode in ('p2_relaxed', 'min_violation_fallback'):
        prior.append(dict(p4=p4, p3=1.))
    if safety.mode == 'min_violation_fallback':
        prior.append(dict(p4=p4, p3=1., p2=1.))
    for candidate in prior:
        if suffix_reachability(*reference_graph(case, **candidate))[0].any():
            raise AssertionError('Recovery skipped an independently feasible higher-priority context')
    forced = None
    if safety.mode == 'min_violation_fallback':
        forced = fallback_path(case, root, edges)
        if tuple(safety.forced_joint_action) != forced:
            raise AssertionError('Fallback is not the independent minimum-violation path')
    return root, edges, root_mask, conditional, forced


class MaskAudit:
    def __init__(self, directory, print_every=100):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.counts = defaultdict(Counter)
        self.modes = Counter()
        self.checked_steps = 0
        self.print_every = print_every
        self.current_case = None
        self.current_step = 0
        self.current_mode = None
        self.csv_file = (self.directory / 'progress.csv').open('w', newline='')
        self.csv = csv.DictWriter(self.csv_file, fieldnames=[
            'checked_steps', 'date', 'mode', 'layer', 'reservoir',
            'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall',
            'false_allow_rate', 'false_block_rate'])
        self.csv.writeheader()

    def compare(self, predicted, expected, layer, rid):
        count = confusion(predicted, expected)
        self.counts[(self.current_mode, layer, rid)].update(count)
        if count['fp'] or count['fn']:
            bad = np.argwhere(np.asarray(predicted, dtype=bool) != expected)
            details = dict(step=self.current_step, mode=self.current_mode, layer=layer,
                           reservoir=rid, counts=count, indices=bad[:30].tolist(),
                           case=self.current_case)
            (self.directory / 'first_mismatch.json').write_text(json.dumps(details, indent=2))
            np.savez_compressed(self.directory / 'first_mismatch_masks.npz',
                                predicted=predicted, expected=expected)
            raise AssertionError(f'Mask mismatch: {layer}/{rid}: FP={count["fp"]}, FN={count["fn"]}')

    def before_sample(self, env, context, step):
        case = make_case(env, context)
        safety = context['safety_result']
        self.current_case, self.current_step, self.current_mode = case, step, safety.mode
        root, edges, expected_root, conditional, forced = reference_masks(case, safety)
        graph = safety.extendability
        order = env.reservoir_order
        self.compare(graph.root_local_mask, root, 'S201_root', order[0])
        for i, matrix in enumerate(graph.compatibility_matrices):
            self.compare(matrix, edges[i], 'S202_all_pairs', order[i+1])
        self.compare(graph.root_action_mask, expected_root, 'S203_root', order[0])
        for i in range(1, len(order)):
            # Every upstream index, not only the action chosen in this rollout.
            predicted = np.stack([graph.get_action_mask(order[i], upstream)
                                  for upstream in range(edges[i-1].shape[0])])
            self.compare(predicted, conditional[i-1], 'S203_all_conditions', order[i])
        if forced is not None:
            expected_root = np.arange(len(root)) == forced[0]
            conditional = [np.broadcast_to(np.arange(m.shape[1]) == forced[i+1], m.shape)
                           for i, m in enumerate(edges)]
        self.compare(safety.get_action_mask(order[0]), expected_root, 'S3_behavior_root', order[0])
        for i in range(1, len(order)):
            predicted = np.stack([safety.get_action_mask(order[i], upstream)
                                  for upstream in range(edges[i-1].shape[0])])
            self.compare(predicted, conditional[i-1], 'S3_behavior_conditions', order[i])
        return expected_root, conditional

    def after_sample(self, sampling, reference):
        root, conditional = reference
        actions = sampling.actions[0, :, 0]
        for i, rid in enumerate(sampling.reservoir_order):
            expected = root if i == 0 else conditional[i-1][actions[i-1]]
            self.compare(sampling.action_masks[i][0], expected, 'S4_actual_mask', rid)
            if not expected[actions[i]]:
                raise AssertionError('Sampled action is not independently executable')
        self.checked_steps += 1
        self.modes[self.current_mode] += 1
        if self.checked_steps % self.print_every == 0:
            self.write('running')

    def write(self, status, error=None):
        combined = Counter()
        records = []
        for (mode, layer, rid), count in sorted(self.counts.items()):
            combined.update(count)
            record = dict(mode=mode, layer=layer, reservoir=rid, **count, **rates(count))
            records.append(record)
            self.csv.writerow(dict(checked_steps=self.checked_steps,
                                   date=self.current_case['date'] if self.current_case else '', **record))
        self.csv_file.flush()
        summary = dict(status=status, checked_steps=self.checked_steps,
                       mode_days=dict(self.modes), uncovered_modes=[m for m in MODES if not self.modes[m]],
                       totals=dict(combined), rates=rates(combined), groups=records,
                       error=error, scope='Visited states and configured discrete actions only; not a proof for all states.')
        temporary = self.directory / 'summary.tmp'
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False))
        temporary.replace(self.directory / 'summary.json')
        accuracy = rates(combined)['accuracy']
        label = 'N/A' if accuracy is None else f'{accuracy:.8%}'
        print(f'[MASK] steps={self.checked_steps} accuracy={label} '
              f'FP={combined["fp"]} FN={combined["fn"]} modes={dict(self.modes)} status={status}', flush=True)

    def close(self):
        self.csv_file.close()


class MaskAccuracyRunner(CascadeReservoirRunner):
    """Audit hooks leave all training and actual sampling in the production runner."""
    def __init__(self, args, algo, env):
        super().__init__(args, algo, env)
        self.mask_audit = MaskAudit(Path(self.run_dir) / 'mask_accuracy',
                                    algo['mask_accuracy']['print_every'])
        self.audit_step = 0

    def collect(self, step):
        context = self.raw_train_env.prepare_step()
        reference = self.mask_audit.before_sample(self.raw_train_env, context, self.audit_step)
        result = super().collect(step)
        self.mask_audit.after_sample(self._pending_sampling_result, reference)
        self.audit_step += 1
        return result

    def train(self):
        result = super().train()
        self.mask_audit.write('running')
        return result

    def run(self):
        try:
            super().run()
        except BaseException as exc:
            if self.mask_audit.current_case is not None:
                (self.mask_audit.directory / 'last_case.json').write_text(
                    json.dumps(self.mask_audit.current_case, indent=2))
            self.mask_audit.write('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', repr(exc))
            raise
        else:
            self.mask_audit.write('completed')

    def close(self):
        try:
            if hasattr(self, 'mask_audit'):
                self.mask_audit.close()
        finally:
            super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load_config', default='harl/configs/experiments/cascade_budget_00001.json')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--exp_name', default='mask_accuracy_200')
    args = parser.parse_args()
    if args.updates < 1 or args.print_every < 1:
        parser.error('updates and print_every must be positive')
    config = json.loads(Path(args.load_config).read_text())
    algo, env = config['algo_args'], config['env_args']
    start, end = [date.fromisoformat(env['simulation'][k]) for k in ('start_date', 'end_date')]
    days = (end-start).days+1
    algo['train'].update(episode_length=days, num_env_steps=days*args.updates,
                         n_rollout_threads=1, model_dir=None, log_interval=1, save_interval=1)
    algo['eval'].update(use_eval=False, n_eval_rollout_threads=1)
    algo['render']['use_render'] = False
    algo['seed'].update(seed=args.seed, seed_specify=True)
    algo['mask_accuracy'] = dict(print_every=args.print_every, every_step=True)
    print(f'Mask experiment: {args.updates} updates x {days} days = {days*args.updates} steps. '
          'All actions/all adjacent pairs checked; stop on any FP or FN.', flush=True)
    runner = MaskAccuracyRunner(dict(algo='happo', env='cascade_reservoir',
                               exp_name=args.exp_name, load_config=args.load_config), algo, env)
    print(f'Mask reports: {runner.mask_audit.directory}', flush=True)
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == '__main__':
    main()
