"""Independent reference labels for the current cascade mask experiment.

No S201/S202/S203, ConstraintContext.resolve, or S3 solver calls are used here.
Shared inputs: configured physical data, Z-V anchors, current state, and reported
recovery parameters. Labels use forward water balance and graph reachability.
"""
from collections import deque

import numpy as np


FLOW_TOLERANCE = 1e-7  # m3/s, the documented numerical contract of S201.
LEVEL_TOLERANCE = 1e-8  # m, operational violation comparison contract.
MODES = ('normal', 'p4_relaxed', 'p3_relaxed', 'p2_relaxed', 'min_violation_fallback')


def make_case(env, context):
    """Read only raw data/state; do not consume production masks as labels."""
    reservoirs = []
    stage = context['operation_stage']
    for i, rid in enumerate(env.reservoir_order):
        physics, mapper = env.physics[rid], env.action_mappers[rid]
        limits = env.level_limits[rid]
        q = mapper.map_min_release_m3s + (
            mapper.map_max_release_m3s - mapper.map_min_release_m3s
        ) * np.linspace(0., 1., mapper.num_actions)
        q[0], q[-1] = mapper.map_min_release_m3s, mapper.map_max_release_m3s
        if not np.allclose(q, mapper.candidate_releases_m3s, rtol=0, atol=FLOW_TOLERANCE):
            raise ValueError('Action mapper differs from the configured linear mapping')
        reservoirs.append(dict(
            id=rid, storage=float(env.storage_m3[rid]),
            previous=env.previous_release_m3s[rid],
            forcing=float(context['external_inflow_m3s'] if i == 0 else context['interval_inflows_m3s'][rid]),
            levels=physics.level_anchors_m.tolist(), volumes=physics.storage_anchors_m3.tolist(),
            releases=q.tolist(),
            p1_low=float(limits['hard_min_level_m']), p1_high=float(limits['safety_upper_level_m']),
            qmin=0., qmax=float(physics.max_total_release_m3s),
            p2_low=float(limits['hard_min_level_m']),
            p2_high=float(limits[env.env_args['level_control'][stage]])))
    return dict(date=context['date'], stage=stage, dt=float(env.timestep_seconds),
                safety=env.env_args['safety'], reservoirs=reservoirs)


def effective_parameters(case, result):
    mode = result.mode
    if mode not in MODES:
        raise ValueError('Unknown recovery mode')
    cfg = case['safety']
    p3, p2 = float(result.p3_relaxation_fraction), float(result.p2_relaxation_fraction)
    if not (0 <= p3 <= 1 and 0 <= p2 <= 1):
        raise ValueError('Relaxation fraction outside [0,1]')
    p4 = result.p4_release_inflow_ratio
    active = case['stage'] == 'dry_supply'
    if active:
        if p4 is None or not cfg['p4']['min_release_inflow_ratio'] <= p4 <= cfg['p4']['normal_release_inflow_ratio']:
            raise ValueError('Invalid active P4 recovery ratio')
    elif p4 is not None:
        raise ValueError('P4 active outside dry_supply')
    if mode == 'normal' and (p3 != 0 or p2 != 0 or (active and p4 != cfg['p4']['normal_release_inflow_ratio'])):
        raise ValueError('Normal mode contains relaxation')
    if mode == 'p4_relaxed' and (not active or p3 != 0 or p2 != 0):
        raise ValueError('P4 mode altered higher-priority constraints')
    if mode == 'p3_relaxed' and p2 != 0:
        raise ValueError('P3 mode altered P2')
    if mode in ('p3_relaxed', 'p2_relaxed', 'min_violation_fallback') and active:
        if p4 != cfg['p4']['min_release_inflow_ratio']:
            raise ValueError('Higher-tier relaxation before P4 maximum')
    if mode in ('p2_relaxed', 'min_violation_fallback') and p3 != 1:
        raise ValueError('P2/fallback before P3 maximum')
    fallback = mode == 'min_violation_fallback'
    if fallback and p2 != 1:
        raise ValueError('Fallback before P2 maximum')
    if fallback != (result.forced_joint_action is not None):
        raise ValueError('Forced path does not match recovery mode')
    return dict(p2=p2, p3=p3, p4=p4, p1_only=fallback)


def local_labels(case, i, inflows, *, p2=0., p3=0., p4=None, p1_only=False):
    """Every candidate release is tested by its resulting storage, not Q_safe."""
    r, cfg, dt = case['reservoirs'][i], case['safety'], case['dt']
    inflow = np.asarray(inflows, dtype=np.float64).reshape(-1, 1)
    q = np.asarray(r['releases'], dtype=np.float64)[None, :]
    next_v = r['storage'] + (inflow - q) * dt
    lo, hi = r['p1_low'], r['p1_high']
    qlo, qhi = np.full_like(inflow, r['qmin']), np.full_like(inflow, r['qmax'])
    if not p1_only:
        lo = max(lo, r['p2_low'] - p2 * cfg['p2']['max_lower_relaxation_m'][r['id']])
        hi = min(hi, r['p2_high'] + p2 * cfg['p2']['max_upper_relaxation_m'][r['id']])
        factor = 1 + p3 * (cfg['p3']['max_relaxation_factor'] - 1)
        h = float(np.interp(r['storage'], r['volumes'], r['levels']))
        delta = cfg['p3']['level_change_limit_m'][r['id']] * factor
        lo, hi = max(lo, h-delta), min(hi, h+delta)
        if r['previous'] is not None:
            dq = factor * max(cfg['p3']['release_change_ratio'] * r['previous'],
                              cfg['p3']['release_change_floor_m3s'])
            qlo = np.maximum(qlo, max(0., r['previous']-dq))
            qhi = np.minimum(qhi, r['previous']+dq)
        if p4 is not None:
            qlo = np.maximum(qlo, p4 * inflow)
    if lo > hi:
        return np.zeros(next_v.shape, dtype=bool)
    vlo, vhi = np.interp([lo, hi], r['levels'], r['volumes'])
    # Numerical tolerances only; this does not clip or execute any action.
    valid = ((next_v >= vlo - dt*FLOW_TOLERANCE) & (next_v <= vhi + dt*FLOW_TOLERANCE)
             & (q >= qlo-FLOW_TOLERANCE) & (q <= qhi+FLOW_TOLERANCE))
    # Empty intersections remain empty, even if individually tolerance-close.
    attainable_low = r['storage'] + (inflow-qhi)*dt
    attainable_high = r['storage'] + (inflow-qlo)*dt
    possible = ((qlo <= qhi) & (np.maximum(vlo, attainable_low)
                               <= np.minimum(vhi, attainable_high) + dt*FLOW_TOLERANCE))
    return valid & possible


def reference_graph(case, **parameters):
    root = local_labels(case, 0, [case['reservoirs'][0]['forcing']], **parameters)[0]
    edges = []
    for i in range(1, len(case['reservoirs'])):
        inflows = np.asarray(case['reservoirs'][i-1]['releases']) + case['reservoirs'][i]['forcing']
        edges.append(local_labels(case, i, inflows, **parameters))
    return root, edges


def suffix_reachability(root, edges):
    """Reverse graph traversal independent of the production S203 recurrence."""
    counts = [len(root)] + [m.shape[1] for m in edges]
    reachable = [np.zeros(n, dtype=bool) for n in counts]
    queue = deque((len(counts)-1, a) for a in range(counts[-1]))
    reachable[-1][:] = True
    while queue:
        layer, action = queue.popleft()
        if layer == 0:
            continue
        parents = np.flatnonzero(edges[layer-1][:, action] & ~reachable[layer-1])
        reachable[layer-1][parents] = True
        queue.extend((layer-1, int(a)) for a in parents)
    return root & reachable[0], [m & reachable[i+1][None, :] for i, m in enumerate(edges)]


def operational_scores(case, i, inflows):
    r, cfg = case['reservoirs'][i], case['safety']
    q = np.asarray(r['releases'])[None, :]
    inflow = np.asarray(inflows).reshape(-1, 1)
    v = r['storage'] + (inflow-q)*case['dt']
    h = np.interp(v, r['volumes'], r['levels'])
    current = np.interp(r['storage'], r['volumes'], r['levels'])
    excess = lambda x, tol: np.where(x > tol, x, 0.)
    p2 = (excess(r['p2_low']-h, LEVEL_TOLERANCE) + excess(h-r['p2_high'], LEVEL_TOLERANCE)) / (r['p1_high']-r['p1_low'])
    dh = cfg['p3']['level_change_limit_m'][r['id']]
    p3 = excess(abs(h-current)-dh, LEVEL_TOLERANCE)/dh
    if r['previous'] is not None:
        dq = max(cfg['p3']['release_change_ratio']*r['previous'], cfg['p3']['release_change_floor_m3s'])
        p3 = p3 + excess(abs(q-r['previous'])-dq, FLOW_TOLERANCE)/dq
    requirement = cfg['p4']['normal_release_inflow_ratio'] * inflow
    p4 = np.zeros(v.shape)
    if case['stage'] == 'dry_supply':
        np.divide(excess(requirement-q, FLOW_TOLERANCE), requirement, out=p4, where=requirement > 0)
    return np.stack([p2, p3, p4], axis=-1)


def fallback_path(case, root, edges):
    """Independent explicit label-setting on the P1 DAG; lexicographic ties."""
    expected_config = dict(comparison='lexicographic_p2_p3_p4', reference='normal_bounds',
        p2_normalization='p1_level_span', p3_normalization='normal_change_limits',
        p4_normalization='normal_required_release', tie_break='joint_action_lexicographic')
    if case['safety']['fallback'] != expected_config:
        raise ValueError('Unsupported fallback comparison schema; update the oracle explicitly')
    values = operational_scores(case, 0, [case['reservoirs'][0]['forcing']])[0]
    frontier = {int(a): (tuple(values[a]), (int(a),)) for a in np.flatnonzero(root)}
    for i, matrix in enumerate(edges):
        inflows = np.asarray(case['reservoirs'][i]['releases']) + case['reservoirs'][i+1]['forcing']
        scores = operational_scores(case, i+1, inflows)
        following = {}
        for parent, (cost, path) in frontier.items():
            for child in np.flatnonzero(matrix[parent]):
                candidate = (tuple(np.asarray(cost)+scores[parent, child]), path+(int(child),))
                if child not in following or candidate < following[child]:
                    following[int(child)] = candidate
        frontier = following
    if not frontier:
        raise ValueError('Reference graph has no P1-safe complete path')
    return min(frontier.values())[1]


def confusion(predicted, expected):
    predicted, expected = np.asarray(predicted), np.asarray(expected, dtype=bool)
    if predicted.shape != expected.shape or not np.all((predicted == 0) | (predicted == 1)):
        raise ValueError('Mask shape or binary encoding is invalid')
    p = predicted.astype(bool)
    return dict(tp=int(np.sum(p & expected)), tn=int(np.sum(~p & ~expected)),
                fp=int(np.sum(p & ~expected)), fn=int(np.sum(~p & expected)))


def rates(counts):
    tp, tn, fp, fn = (counts[k] for k in ('tp', 'tn', 'fp', 'fn'))
    divide = lambda a, b: a/b if b else None
    return dict(accuracy=divide(tp+tn, tp+tn+fp+fn), precision=divide(tp, tp+fp),
                recall=divide(tp, tp+fn), false_allow_rate=divide(fp, tn+fp),
                false_block_rate=divide(fn, tp+fn))
