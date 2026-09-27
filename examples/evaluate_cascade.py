"""Compare two frozen checkpoints over their complete configured period.

Run from the repository root: python -m examples.evaluate_cascade --help
"""
import argparse
import copy
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


def read_checkpoint(directory):
    directory = Path(directory).resolve()
    required = ["config.json", "checkpoint_metadata.json", "actor_observation_encoding.json",
                "critic_agent.pt", "constraint_critics.pt", "lagrangian_state.npz"]
    required += [f"actor_agent{i}.pt" for i in range(5)]
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {directory / name}")
    CascadeReservoirRunner._validate_observation_checkpoint(directory)
    config = json.loads((directory / "config.json").read_text())
    if config["algo_args"]["train"]["use_valuenorm"] and not (directory / "value_normalizer.pt").is_file():
        raise FileNotFoundError(f"Missing value normalizer: {directory}")
    metadata = json.loads((directory / "checkpoint_metadata.json").read_text())
    if metadata["kind"] not in ("run_start", "periodic", "final"):
        raise ValueError("Unknown checkpoint kind")
    return directory, config, metadata


def snapshot_checkpoint(checkpoint, destination):
    """Reject a concurrently changing source; evaluate only the private copy."""
    source = checkpoint[0]
    files = sorted(p for p in source.iterdir() if p.is_file())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    destination.mkdir(parents=True)
    for path in files:
        shutil.copyfile(path, destination / path.name)
    for path in files:
        if (hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]
                or hashlib.sha256((destination / path.name).read_bytes()).hexdigest() != hashes[path.name]):
            raise RuntimeError("Checkpoint changed while copying; retry after saving completes")
    snapshot = read_checkpoint(destination)
    if snapshot[1:] != checkpoint[1:]:
        raise RuntimeError("Checkpoint metadata/config changed during evaluation setup")
    (destination / "source_hashes.json").write_text(json.dumps(
        {"source": str(source), "sha256": hashes}, indent=2))
    return snapshot


def parameter_digest(runner):
    digest = hashlib.sha256()
    networks = [a.actor for a in runner.actor] + [runner.critic.critic]
    networks += [c.critic for c in runner.constraint_critic.critics]
    networks += [runner.value_normalizer] + list(runner.constraint_critic.value_normalizers)
    for network in networks:
        if network is not None:
            for key, value in network.state_dict().items():
                digest.update(key.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(runner.lagrangian_manager.snapshot().tobytes())
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cost_statistics(rows, gamma):
    costs = np.array([float(r["cost"]) for r in rows])
    active = np.array([float(r["active"]) for r in rows])
    weights = gamma ** np.arange(len(rows), dtype=np.float64)
    count = int(active.sum())
    return {
        "active_days": count,
        "violation_days": int(np.sum((costs > 0) & (active > 0))),
        "mean_cost": float(costs.sum() / count) if count else None,
        "discounted_mean_cost": float(np.dot(weights, costs) / np.dot(weights, active)) if count else None,
    }


def summarize(run_dir, config):
    system = read_rows(run_dir / "cascade_system_timeseries.csv")
    reservoirs = read_rows(run_dir / "cascade_reservoir_timeseries.csv")
    constraints = read_rows(run_dir / "cascade_constraint_timeseries.csv")
    simulation = config["env_args"]["simulation"]
    start, end = (date.fromisoformat(simulation[k]) for k in ("start_date", "end_date"))
    dates = [(start + timedelta(days=t)).isoformat() for t in range((end-start).days + 1)]
    if [r["date"] for r in system] != dates or any(r["phase"] != "eval" for r in system):
        raise ValueError("Evaluation did not cover exactly one complete configured period")
    if [int(r["done"]) for r in system] != [0] * (len(dates)-1) + [1]:
        raise ValueError("Unexpected episode termination")
    gamma = config["algo_args"]["algo"]["gamma"]
    result = {
        "days": len(dates), "start_date": dates[0], "end_date": dates[-1],
        "reward_sum": sum(float(r["reward"]) for r in system),
        "energy_mwh": sum(float(r["system_energy_mwh"]) for r in system),
        "safety_mode_days": dict(Counter(r["safety_mode"] for r in system)),
        "constraints": {}, "reservoirs": {},
    }
    for spec in config["env_args"]["long_term_constraints"]:
        cid = spec["reservoir_id"] + ":" + spec["type"]
        rows = [r for r in constraints if r["constraint_id"] == cid]
        if [r["date"] for r in rows] != dates:
            raise ValueError(f"Incomplete constraint trajectory: {cid}")
        stats = cost_statistics(rows, gamma)
        stats["budget"] = spec["budget"]
        stats["yearly"] = {year: cost_statistics([r for r in rows if r["date"].startswith(year)], gamma)
                           for year in sorted({d[:4] for d in dates})}
        # Annual discount weights restart within each year: diagnostic, not the training statistic.
        result["constraints"][cid] = stats
    for rid in config["env_args"]["reservoir_order"]:
        rows = [r for r in reservoirs if r["reservoir_id"] == rid]
        if [r["date"] for r in rows] != dates:
            raise ValueError(f"Incomplete reservoir trajectory: {rid}")
        result["reservoirs"][rid] = {
            "end_storage_m3": float(rows[-1]["next_storage_m3"]),
            "end_level_m": float(rows[-1]["next_level_m"]),
            "max_p2_upper_violation_m": max(float(r["p2_upper_violation_m"]) for r in rows),
            "max_p2_lower_violation_m": max(float(r["p2_lower_violation_m"]) for r in rows),
        }
    return result


def evaluate_checkpoint(checkpoint, output, label):
    directory, original_config, metadata = checkpoint
    config = copy.deepcopy(original_config)
    algo = config["algo_args"]
    algo["train"]["model_dir"] = str(directory)
    algo["train"]["n_rollout_threads"] = 1
    algo["eval"].update(use_eval=True, eval_episodes=1, n_eval_rollout_threads=1)
    algo["render"]["use_render"] = False
    algo["device"]["cuda"] = False
    algo["logger"]["log_dir"] = str(output / label)
    args = dict(algo="happo", env="cascade_reservoir", exp_name="frozen_evaluation")
    runner = CascadeReservoirRunner(args, algo, config["env_args"])
    try:
        before = parameter_digest(runner)
        runner.logger.init(0)
        runner.logger.episode_init(0)
        runner.prep_rollout()
        runner.eval()
        if parameter_digest(runner) != before:
            raise RuntimeError("Evaluation changed model parameters or lambda")
        result = summarize(Path(runner.run_dir), original_config)
        result.update(checkpoint=str(directory), metadata=metadata,
                      parameter_sha256=before, logs=str(runner.run_dir))
        return result
    finally:
        runner.close()


def compare(model_dir, baseline_dir, output_dir):
    trained = read_checkpoint(model_dir)
    baseline = read_checkpoint(baseline_dir)
    for key in ("env_args",):
        if trained[1][key] != baseline[1][key]:
            raise ValueError("Baseline and trained environment configurations differ")
    for key in ("model", "algo"):
        if trained[1]["algo_args"][key] != baseline[1]["algo_args"][key]:
            raise ValueError(f"Baseline and trained {key} configurations differ")
    if trained[2]["kind"] == "run_start" or baseline[2]["kind"] != "run_start":
        raise ValueError("Expected trained periodic/final and baseline run_start checkpoints")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = [(label, snapshot_checkpoint(checkpoint, output / "checkpoints" / label))
                 for label, checkpoint in [("baseline", baseline), ("trained", trained)]]
    results = {}
    for label, checkpoint in snapshots:
        print(f"Evaluating {label}: {checkpoint[0]}", flush=True)
        results[label] = evaluate_checkpoint(checkpoint, output, label)
        (output / f"{label}.json").write_text(json.dumps(results[label], indent=2, allow_nan=False))
    b, t = results["baseline"], results["trained"]
    results["energy_change_percent"] = 100 * (t["energy_mwh"] / b["energy_mwh"] - 1) if b["energy_mwh"] else None
    results["selection"] = "deterministic argmax under sequential safety masks"
    (output / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    lines = ["# 固定策略完整周期对照", "", "相同历史区间、确定性动作选择；不代表持出情景泛化能力。", "",
             "| 指标 | 起始策略 | 训练后策略 |", "|---|---:|---:|",
             f"| 发电量 MWh | {b['energy_mwh']:.3f} | {t['energy_mwh']:.3f} |",
             f"| 累计 reward | {b['reward_sum']:.6f} | {t['reward_sum']:.6f} |"]
    for cid in b["constraints"]:
        for metric in ("mean_cost", "discounted_mean_cost", "violation_days"):
            lines.append(f"| {cid} {metric} | {b['constraints'][cid][metric]} | {t['constraints'][cid][metric]} |")
    for rid in b["reservoirs"]:
        lines.append(f"| {rid} 末期库容 m³ | {b['reservoirs'][rid]['end_storage_m3']} | {t['reservoirs'][rid]['end_storage_m3']} |")
    lines += ["", f"发电量变化：{results['energy_change_percent']}%。", "",
              "年度统计、恢复模式、P2 违反量和检查点信息见 comparison.json；年度折扣从各年起点计算，仅作诊断。",
              "cost 不是缺水百分比；发电增加不自动代表约束改善。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", type=Path, help="Defaults to MODEL_DIR/initial")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    args = parser.parse_args()
    compare(args.model_dir, args.baseline_dir or args.model_dir / "initial", args.output_dir)
    print(f"Evaluation report: {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
