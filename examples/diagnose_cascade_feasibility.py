"""Replay logged actions and audit ecology choices without changing training.

This is a state-conditional audit, NOT a proof of trajectory-level feasibility.
Run from the project root with --run-dir and a new --output-dir.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv


def path_nodes(root, matrices, allowed):
    """Nodes on complete paths, requiring every node to satisfy allowed.

    Forward reachability alone includes dead ends; backward reachability alone
    includes nodes that cannot be reached from the root. Both are necessary.
    """
    forward = [np.asarray(root, dtype=bool) & allowed[0]]
    for i, matrix in enumerate(matrices):
        forward.append(np.any(matrix & forward[-1][:, None], axis=0) & allowed[i+1])
    backward = [None] * len(allowed)
    backward[-1] = allowed[-1].copy()
    for i in range(len(matrices)-1, -1, -1):
        backward[i] = allowed[i] & np.any(matrices[i] & backward[i+1][None, :], axis=1)
    return [a & b for a, b in zip(forward, backward)]


def executable_nodes(safety, allowed):
    if safety.forced_joint_action is not None:
        # The Actor cannot select arbitrary P1 paths in final fallback.
        allowed = [a & (np.arange(len(a)) == index)
                   for a, index in zip(allowed, safety.forced_joint_action)]
    graph = safety.extendability
    return path_nodes(graph.root_local_mask, graph.compatibility_matrices, allowed)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit(run_dir, output_dir, update=None):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    config_text = (run_dir / "config.json").read_text()
    config = json.loads(config_text)
    if update is None:
        update = max(int(r["update"]) for r in read_rows(run_dir / "cascade_lagrangian_updates.csv"))
    rows = [r for r in read_rows(run_dir / "cascade_reservoir_timeseries.csv")
            if r["phase"] == "train" and int(r["update"]) == update]
    constraints = [r for r in read_rows(run_dir / "cascade_constraint_timeseries.csv")
                   if r["phase"] == "train" and int(r["update"]) == update]
    env_args = config["env_args"]
    order = env_args["reservoir_order"]
    specs = env_args["long_term_constraints"]
    if len(specs) != len(order) or any(s["type"] != "eco_release" for s in specs):
        raise ValueError("This audit requires one ecology constraint per reservoir")
    specs = {s["reservoir_id"]: s for s in specs}
    if set(specs) != set(order):
        raise ValueError("Duplicate or missing reservoir constraints")
    if any(s["tolerance"] != 0 for s in specs.values()):
        raise ValueError("This strict-ecology audit currently requires zero cost tolerance")
    start, end = [date.fromisoformat(env_args["simulation"][k]) for k in ("start_date", "end_date")]
    dates = [str(start + timedelta(days=t)) for t in range((end-start).days+1)]
    by_day = {(r["date"], r["reservoir_id"]): r for r in rows}
    costs = {(r["date"], r["constraint_id"]): r for r in constraints}
    expected = {(d, rid) for d in dates for rid in order}
    if len(rows) != len(expected) or set(by_day) != expected:
        raise ValueError("Replay requires exactly one complete configured period")
    if len(costs) != len(expected) or len(constraints) != len(expected):
        raise ValueError("Incomplete or duplicate constraint records")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_config.json").write_text(config_text)
    for name, data in (("source_reservoir_rows.json", rows), ("source_constraint_rows.json", constraints)):
        (output_dir / name).write_text(json.dumps(data, ensure_ascii=False))
    env = CascadeReservoirEnv(env_args)
    env.reset()
    results = []
    simultaneous_days = 0
    try:
        for t, day in enumerate(dates):
            context = env.prepare_step()
            if context["date"] != day:
                raise ValueError("Date mismatch during replay")
            safety = context["safety_result"]
            candidates = [env.action_mappers[rid].candidate_releases_m3s for rid in order]
            actions = []
            for rid, q in zip(order, candidates):
                release = float(by_day[day, rid]["release_m3s"])
                matches = np.flatnonzero(np.isclose(q, release, rtol=0, atol=1e-7))
                if len(matches) != 1:
                    raise ValueError("Logged release does not uniquely match an action")
                actions.append(int(matches[0]))
            allowed = [np.ones(len(q), dtype=bool) for q in candidates]
            nodes = executable_nodes(safety, allowed)
            ecological = []
            for rid, q in zip(order, candidates):
                active = bool(int(float(costs[day, rid+":eco_release"]["active"])))
                ecological.append(q >= specs[rid]["requirement"] if active else np.ones(len(q), dtype=bool))
            simultaneous = bool(np.any(executable_nodes(safety, ecological)[0]))
            simultaneous_days += int(simultaneous)
            for i, rid in enumerate(order):
                spec = specs[rid]
                logged_cost = costs[day, rid+":eco_release"]
                mask = safety.get_action_mask(rid, None if i == 0 else actions[i-1])
                if not mask[actions[i]] or not nodes[i][actions[i]]:
                    raise ValueError("Logged action not on executable path")
                if safety.mode != by_day[day, rid]["safety_mode"]:
                    raise ValueError("Recovery mode differs from original run")
                active = int(float(logged_cost["active"]))
                violation = active and float(logged_cost["cost"]) > 0
                conditional_ok = bool(np.any(mask & ecological[i]))
                joint_ok = bool(np.any(nodes[i] & ecological[i]))
                category = ("satisfied_or_inactive" if not violation else
                            "actor_choice_available" if conditional_ok else
                            "upstream_choice_required" if joint_ok else
                            "no_action_at_logged_state")
                qmax = float(np.max(candidates[i][nodes[i]]))
                minimum = active * max(0, np.sqrt(max(0, (spec["requirement"]-qmax)/spec["requirement"]))
                                       / spec["normalizer"] - spec["tolerance"])
                results.append(dict(date=day, reservoir_id=rid, mode=safety.mode,
                                    active=active, cost=float(logged_cost["cost"]),
                                    category=category, conditional_action_count=int(mask.sum()),
                                    complete_path_action_count=int(nodes[i].sum()),
                                    conditional_eco_possible=conditional_ok,
                                    joint_eco_possible=joint_ok, all_eco_path_possible=simultaneous,
                                    state_conditional_min_cost=float(minimum)))
            _, _, _, dones, infos, _ = env.step(actions)
            if all(dones) != (t == len(dates)-1):
                raise ValueError("Unexpected replay termination")
            for i, rid in enumerate(order):
                expected_storage = float(by_day[day, rid]["next_storage_m3"])
                np.testing.assert_allclose(env.storage_m3[rid], expected_storage, rtol=0, atol=1e-3)
                j = tuple(infos[0]["constraint_ids"]).index(rid+":eco_release")
                np.testing.assert_allclose(infos[0]["constraint_raw_violations"][j],
                                           float(costs[day, rid+":eco_release"]["raw_violation"]),
                                           rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(infos[0]["constraint_costs"][j],
                                           float(costs[day, rid+":eco_release"]["cost"]),
                                           rtol=1e-6, atol=1e-7)
                if infos[0]["constraint_active_flags"][j] != float(costs[day, rid+":eco_release"]["active"]):
                    raise ValueError("Constraint activation differs during replay")
            if (t+1) % 100 == 0 or t+1 == len(dates):
                print(f"Replay verified: {t+1}/{len(dates)} days", flush=True)
    finally:
        env.close()
    with (output_dir / "daily_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    gamma = config["algo_args"]["algo"]["gamma"]
    weights = gamma ** np.arange(len(dates))
    summary = dict(source=str(run_dir.resolve()), update=update, days=len(dates),
                   config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
                   simultaneous_ecology_path_days=simultaneous_days, reservoirs={})
    for rid in order:
        r = [r for r in results if r["reservoir_id"] == rid]
        denominator = np.dot(weights, [x["active"] for x in r])
        summary["reservoirs"][rid] = dict(
            categories=dict(Counter(x["category"] for x in r)),
            discounted_cost=float(np.dot(weights, [x["cost"] for x in r])/denominator) if denominator else None,
            state_conditional_discounted_min=float(np.dot(weights, [x["state_conditional_min_cost"] for x in r])/denominator) if denominator else None)
    summary["limitation"] = "Minima keep logged states fixed; they are not attainable multi-day policies or global budget lower bounds."
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--update", type=int)
    args = parser.parse_args()
    audit(args.run_dir, args.output_dir, args.update)
