from __future__ import annotations

import csv
import os
from datetime import date, timedelta
from typing import Mapping, Sequence

import numpy as np

from harl.common.base_logger import BaseLogger


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)


class CascadeReservoirLogger(BaseLogger):

    def __init__(
        self,
        args,
        algo_args,
        env_args,
        num_agents,
        writter,
        run_dir,
    ):
        super().__init__(
            args,
            algo_args,
            env_args,
            num_agents,
            writter,
            run_dir,
        )

        self.reservoir_order = tuple(
            env_args["reservoir_order"]
        )

        if (
            self.reservoir_order
            != _RESERVOIR_ORDER
        ):
            raise ValueError(
                "cascade reservoir order must be "
                "WDD, BHT, XLD, XJB, THR"
            )

        if num_agents != len(
            _RESERVOIR_ORDER
        ):
            raise ValueError(
                "CascadeReservoirLogger requires "
                "exactly five agents"
            )

        self.simulation_start_date = (
            date.fromisoformat(
                str(
                    env_args[
                        "simulation"
                    ][
                        "start_date"
                    ]
                )
            )
        )

        self.constraint_ids = tuple(
            (
                f"{item['reservoir_id']}:"
                f"{item['type']}"
            )
            for item
            in env_args[
                "long_term_constraints"
            ]
        )

        if len(
            self.constraint_ids
        ) != len(
            set(
                self.constraint_ids
            )
        ):
            raise ValueError(
                "long-term constraint IDs "
                "must be unique"
            )

        self.constraint_budgets = {
            (
                f"{item['reservoir_id']}:"
                f"{item['type']}"
            ): float(
                item["budget"]
            )
            for item
            in env_args[
                "long_term_constraints"
            ]
        }

        self._open_csv_files()

        self._train_step = 0
        self._eval_step = 0
        self._rollout_step = 0

        self._train_env_day = None
        self._train_env_episode = None

        self._eval_env_day = None
        self._eval_env_episode = None

        self._rollout_metrics = {}
        self._eval_metrics = {}

        self._steps_since_flush = 0

    def get_task_name(self):
        return "-".join(
            self.env_args[
                "reservoir_order"
            ]
        )

    def init(
        self,
        episodes,
    ):
        super().init(
            episodes
        )

        num_threads = int(
            self.algo_args[
                "train"
            ][
                "n_rollout_threads"
            ]
        )

        self._train_env_day = np.zeros(
            num_threads,
            dtype=np.int64,
        )

        self._train_env_episode = np.zeros(
            num_threads,
            dtype=np.int64,
        )

        self._train_step = 0

    def episode_init(
        self,
        episode,
    ):
        super().episode_init(
            episode
        )

        self._rollout_step = 0
        self._rollout_metrics = {}

    def per_step(
        self,
        data,
    ):
        super().per_step(
            data
        )

        (
            _,
            _,
            rewards,
            dones,
            infos,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = data

        self._record_environment_step(
            phase="train",
            rewards=rewards,
            dones=dones,
            infos=infos,
        )

        self._rollout_step += 1

    def episode_log(
        self,
        actor_train_infos,
        critic_train_info,
        actor_buffer,
        critic_buffer,
    ):
        super().episode_log(
            actor_train_infos,
            critic_train_info,
            actor_buffer,
            critic_buffer,
        )

        if self._rollout_metrics:
            env_infos = {
                key: values
                for key, values
                in self._rollout_metrics.items()
                if len(values) > 0
            }

            self.log_env(
                env_infos
            )

        self._write_lagrangian_update(
            critic_train_info
        )

        self._print_cascade_summary()

        self._flush_files()

    def eval_init(self):
        super().eval_init()

        num_threads = int(
            self.algo_args[
                "eval"
            ][
                "n_eval_rollout_threads"
            ]
        )

        self._eval_env_day = np.zeros(
            num_threads,
            dtype=np.int64,
        )

        self._eval_env_episode = np.zeros(
            num_threads,
            dtype=np.int64,
        )

        self._eval_step = 0
        self._eval_metrics = {}

    def eval_per_step(
        self,
        eval_data,
    ):
        super().eval_per_step(
            eval_data
        )

        (
            _,
            _,
            rewards,
            dones,
            infos,
            _,
        ) = eval_data

        self._record_environment_step(
            phase="eval",
            rewards=rewards,
            dones=dones,
            infos=infos,
        )

    def eval_log(
        self,
        eval_episode,
    ):
        super().eval_log(
            eval_episode
        )

        if self._eval_metrics:
            env_infos = {
                f"eval/{key}": values
                for key, values
                in self._eval_metrics.items()
                if len(values) > 0
            }

            self.log_env(
                env_infos
            )

        self._flush_files()

    def close(self):
        self._flush_files()

        self._reservoir_file.close()
        self._system_file.close()
        self._constraint_file.close()
        self._lagrangian_file.close()

        super().close()

    def _record_environment_step(
        self,
        phase,
        rewards,
        dones,
        infos,
    ):
        rewards = np.asarray(
            rewards,
            dtype=np.float64,
        )

        dones = np.asarray(
            dones,
            dtype=bool,
        )

        num_threads = len(
            infos
        )

        if rewards.shape[0] != num_threads:
            raise ValueError(
                "reward thread count does not "
                "match info thread count"
            )

        if dones.shape != (
            num_threads,
            self.num_agents,
        ):
            raise ValueError(
                "dones has invalid shape"
            )

        done_envs = np.all(
            dones,
            axis=1,
        )

        for thread_id in range(
            num_threads
        ):
            thread_infos = infos[
                thread_id
            ]

            if len(
                thread_infos
            ) != self.num_agents:
                raise ValueError(
                    "each thread must provide "
                    "five agent info dictionaries"
                )

            for info in thread_infos:
                if not isinstance(
                    info,
                    Mapping,
                ):
                    raise TypeError(
                        "agent info must be a mapping"
                    )

            first_info = thread_infos[
                0
            ]

            current_date = (
                self._resolve_date(
                    phase=phase,
                    thread_id=thread_id,
                    info=first_info,
                )
            )

            (
                training_step,
                phase_step,
                env_episode,
            ) = self._step_metadata(
                phase,
                thread_id,
            )

            reward = float(
                np.mean(
                    rewards[
                        thread_id
                    ]
                )
            )

            reservoir_values = []

            for (
                agent_id,
                reservoir_id,
            ) in enumerate(
                self.reservoir_order
            ):
                info = thread_infos[
                    agent_id
                ]

                info_reservoir_id = (
                    info.get(
                        "reservoir_id"
                    )
                )

                if (
                    info_reservoir_id
                    is not None
                    and str(
                        info_reservoir_id
                    )
                    != reservoir_id
                ):
                    raise ValueError(
                        "agent info reservoir order "
                        "does not match "
                        "WDD-BHT-XLD-XJB-THR"
                    )

                inflow = self._required_number(
                    info,
                    (
                        "inflow_m3s",
                    ),
                    "inflow",
                )

                release = self._required_number(
                    info,
                    (
                        "executed_release_m3s",
                        "release_m3s",
                    ),
                    "executed release",
                )

                storage = self._required_number(
                    info,
                    (
                        "next_storage_m3",
                    ),
                    "next storage",
                )

                level = self._required_number(
                    info,
                    (
                        "next_level_m",
                    ),
                    "next level",
                )

                power = self._required_number(
                    info,
                    (
                        "power_mw",
                    ),
                    "power",
                )

                energy = self._required_number(
                    info,
                    (
                        "energy_mwh",
                    ),
                    "energy",
                )

                safety_mode = str(
                    info.get(
                        "safety_mode",
                        "",
                    )
                )

                p4_ratio = self._optional_number(
                    info,
                    "p4_release_inflow_ratio",
                )

                p3_fraction = self._optional_number(
                    info,
                    "p3_relaxation_fraction",
                )

                p2_violation = self._optional_number(
                    info,
                    "p2_total_violation",
                )

                reservoir_values.append(
                    {
                        "reservoir_id": reservoir_id,
                        "inflow_m3s": inflow,
                        "release_m3s": release,
                        "storage_m3": storage,
                        "level_m": level,
                        "power_mw": power,
                        "energy_mwh": energy,
                    }
                )

                self._reservoir_writer.writerow(
                    {
                        "phase": phase,
                        "training_step": training_step,
                        "phase_step": phase_step,
                        "update": self.episode,
                        "rollout_step": (
                            self._rollout_step
                            if phase == "train"
                            else ""
                        ),
                        "thread": thread_id,
                        "env_episode": env_episode,
                        "date": current_date,
                        "reservoir_id": reservoir_id,
                        "inflow_m3s": inflow,
                        "release_m3s": release,
                        "next_storage_m3": storage,
                        "next_level_m": level,
                        "power_mw": power,
                        "energy_mwh": energy,
                        "safety_mode": safety_mode,
                        "p4_release_inflow_ratio": (
                            p4_ratio
                        ),
                        "p3_relaxation_fraction": (
                            p3_fraction
                        ),
                        "p2_total_violation": (
                            p2_violation
                        ),
                    }
                )

                metric_store = (
                    self._rollout_metrics
                    if phase == "train"
                    else self._eval_metrics
                )

                self._append_metric(
                    metric_store,
                    (
                        f"operation/"
                        f"{reservoir_id}/"
                        "level_m"
                    ),
                    level,
                )

                self._append_metric(
                    metric_store,
                    (
                        f"operation/"
                        f"{reservoir_id}/"
                        "release_m3s"
                    ),
                    release,
                )

                self._append_metric(
                    metric_store,
                    (
                        f"operation/"
                        f"{reservoir_id}/"
                        "power_mw"
                    ),
                    power,
                )

            system_energy = (
                self._optional_number(
                    first_info,
                    "system_energy_mwh",
                )
            )

            if not np.isfinite(
                system_energy
            ):
                system_energy = float(
                    sum(
                        item[
                            "energy_mwh"
                        ]
                        for item
                        in reservoir_values
                    )
                )

            total_power = float(
                sum(
                    item[
                        "power_mw"
                    ]
                    for item
                    in reservoir_values
                )
            )

            total_storage = float(
                sum(
                    item[
                        "storage_m3"
                    ]
                    for item
                    in reservoir_values
                )
            )

            total_release = float(
                sum(
                    item[
                        "release_m3s"
                    ]
                    for item
                    in reservoir_values
                )
            )

            safety_mode = str(
                first_info.get(
                    "safety_mode",
                    "",
                )
            )

            p4_ratio = self._optional_number(
                first_info,
                "p4_release_inflow_ratio",
            )

            p3_fraction = self._optional_number(
                first_info,
                "p3_relaxation_fraction",
            )

            p2_violation = self._optional_number(
                first_info,
                "p2_total_violation",
            )

            self._system_writer.writerow(
                {
                    "phase": phase,
                    "training_step": training_step,
                    "phase_step": phase_step,
                    "update": self.episode,
                    "rollout_step": (
                        self._rollout_step
                        if phase == "train"
                        else ""
                    ),
                    "thread": thread_id,
                    "env_episode": env_episode,
                    "date": current_date,
                    "reward": reward,
                    "system_energy_mwh": (
                        system_energy
                    ),
                    "total_power_mw": (
                        total_power
                    ),
                    "total_storage_m3": (
                        total_storage
                    ),
                    "total_release_m3s": (
                        total_release
                    ),
                    "safety_mode": (
                        safety_mode
                    ),
                    "p4_release_inflow_ratio": (
                        p4_ratio
                    ),
                    "p3_relaxation_fraction": (
                        p3_fraction
                    ),
                    "p2_total_violation": (
                        p2_violation
                    ),
                    "done": int(
                        done_envs[
                            thread_id
                        ]
                    ),
                }
            )

            metric_store = (
                self._rollout_metrics
                if phase == "train"
                else self._eval_metrics
            )

            self._append_metric(
                metric_store,
                "system/reward",
                reward,
            )

            self._append_metric(
                metric_store,
                "system/energy_mwh",
                system_energy,
            )

            self._append_metric(
                metric_store,
                "system/total_power_mw",
                total_power,
            )

            self._append_metric(
                metric_store,
                "safety/p2_total_violation",
                p2_violation,
            )

            self._append_metric(
                metric_store,
                "safety/p3_relaxation_fraction",
                p3_fraction,
            )

            self._append_metric(
                metric_store,
                "safety/p4_release_inflow_ratio",
                p4_ratio,
            )

            self._record_constraints(
                phase=phase,
                training_step=training_step,
                phase_step=phase_step,
                thread_id=thread_id,
                env_episode=env_episode,
                current_date=current_date,
                info=first_info,
                metric_store=metric_store,
            )

            self._advance_environment_clock(
                phase=phase,
                thread_id=thread_id,
                done=bool(
                    done_envs[
                        thread_id
                    ]
                ),
            )

        if phase == "train":
            self._train_step += (
                num_threads
            )
        else:
            self._eval_step += (
                num_threads
            )

        self._steps_since_flush += (
            num_threads
        )

        if self._steps_since_flush >= 64:
            self._flush_files()

    def _record_constraints(
        self,
        phase,
        training_step,
        phase_step,
        thread_id,
        env_episode,
        current_date,
        info,
        metric_store,
    ):
        required_keys = (
            "constraint_ids",
            "constraint_costs",
            "constraint_active_flags",
        )

        missing = [
            key
            for key
            in required_keys
            if key not in info
        ]

        if missing:
            raise KeyError(
                "environment info missing "
                "long-term constraint fields: "
                f"{missing}"
            )

        constraint_ids = tuple(
            str(value)
            for value
            in info[
                "constraint_ids"
            ]
        )

        if (
            constraint_ids
            != self.constraint_ids
        ):
            raise ValueError(
                "logger constraint IDs do not "
                "match environment constraint IDs"
            )

        costs = self._constraint_vector(
            info[
                "constraint_costs"
            ],
            "constraint_costs",
        )

        active_flags = (
            self._constraint_vector(
                info[
                    "constraint_active_flags"
                ],
                "constraint_active_flags",
            )
        )

        if not np.all(
            (
                active_flags == 0.0
            )
            | (
                active_flags == 1.0
            )
        ):
            raise ValueError(
                "constraint active flags "
                "must be binary"
            )

        raw_violations = (
            self._optional_constraint_vector(
                info.get(
                    "constraint_raw_violations"
                )
            )
        )

        normalized_violations = (
            self._optional_constraint_vector(
                info.get(
                    "constraint_normalized_violations"
                )
            )
        )

        for index, constraint_id in enumerate(
            self.constraint_ids
        ):
            active = float(
                active_flags[
                    index
                ]
            )

            cost = float(
                costs[
                    index
                ]
            )

            self._constraint_writer.writerow(
                {
                    "phase": phase,
                    "training_step": training_step,
                    "phase_step": phase_step,
                    "update": self.episode,
                    "rollout_step": (
                        self._rollout_step
                        if phase == "train"
                        else ""
                    ),
                    "thread": thread_id,
                    "env_episode": env_episode,
                    "date": current_date,
                    "constraint_id": constraint_id,
                    "raw_violation": (
                        raw_violations[
                            index
                        ]
                    ),
                    "normalized_violation": (
                        normalized_violations[
                            index
                        ]
                    ),
                    "cost": cost,
                    "active": int(
                        active
                    ),
                    "budget": (
                        self.constraint_budgets[
                            constraint_id
                        ]
                    ),
                }
            )

            if active > 0.0:
                self._append_metric(
                    metric_store,
                    (
                        f"constraint/"
                        f"{constraint_id}/cost"
                    ),
                    cost,
                )

    def _write_lagrangian_update(
        self,
        critic_train_info,
    ):
        for constraint_id in (
            self.constraint_ids
        ):
            lambda_key = (
                f"lambda/{constraint_id}"
            )

            if lambda_key not in (
                critic_train_info
            ):
                continue

            mean_cost_key = (
                f"mean_cost/{constraint_id}"
            )

            budget_key = (
                f"budget/{constraint_id}"
            )

            gap_key = (
                f"budget_gap/{constraint_id}"
            )

            active_key = (
                f"active/{constraint_id}"
            )

            self._lagrangian_writer.writerow(
                {
                    "update": self.episode,
                    "training_step": (
                        self.total_num_steps
                    ),
                    "constraint_id": (
                        constraint_id
                    ),
                    "lambda": float(
                        critic_train_info[
                            lambda_key
                        ]
                    ),
                    "mean_cost": (
                        float(
                            critic_train_info[
                                mean_cost_key
                            ]
                        )
                        if mean_cost_key
                        in critic_train_info
                        else ""
                    ),
                    "budget": float(
                        critic_train_info.get(
                            budget_key,
                            self.constraint_budgets[
                                constraint_id
                            ],
                        )
                    ),
                    "budget_gap": (
                        float(
                            critic_train_info[
                                gap_key
                            ]
                        )
                        if gap_key
                        in critic_train_info
                        else ""
                    ),
                    "active_in_batch": int(
                        float(
                            critic_train_info.get(
                                active_key,
                                0.0,
                            )
                        )
                        > 0.5
                    ),
                }
            )

    def _print_cascade_summary(
        self,
    ):
        energy_values = (
            self._rollout_metrics.get(
                "system/energy_mwh",
                [],
            )
        )

        reward_values = (
            self._rollout_metrics.get(
                "system/reward",
                [],
            )
        )

        p2_values = (
            self._rollout_metrics.get(
                "safety/p2_total_violation",
                [],
            )
        )

        if energy_values:
            print(
                "Cascade rollout: "
                f"mean system energy = "
                f"{np.mean(energy_values):.3f} MWh, "
                f"mean reward = "
                f"{np.mean(reward_values):.6f}, "
                f"mean P2 violation = "
                f"{np.nanmean(p2_values):.6f}."
            )

    def _step_metadata(
        self,
        phase,
        thread_id,
    ):
        if phase == "train":
            return (
                self._train_step
                + thread_id,
                self._train_step
                + thread_id,
                int(
                    self._train_env_episode[
                        thread_id
                    ]
                ),
            )

        return (
            int(
                getattr(
                    self,
                    "total_num_steps",
                    0,
                )
            ),
            self._eval_step
            + thread_id,
            int(
                self._eval_env_episode[
                    thread_id
                ]
            ),
        )

    def _resolve_date(
        self,
        phase,
        thread_id,
        info,
    ):
        for key in (
            "date",
            "executed_date",
        ):
            if key in info:
                return str(
                    info[
                        key
                    ]
                )[
                    :10
                ]

        if phase == "train":
            day_index = int(
                self._train_env_day[
                    thread_id
                ]
            )
        else:
            day_index = int(
                self._eval_env_day[
                    thread_id
                ]
            )

        return str(
            self.simulation_start_date
            + timedelta(
                days=day_index
            )
        )

    def _advance_environment_clock(
        self,
        phase,
        thread_id,
        done,
    ):
        if phase == "train":
            day_array = (
                self._train_env_day
            )

            episode_array = (
                self._train_env_episode
            )
        else:
            day_array = (
                self._eval_env_day
            )

            episode_array = (
                self._eval_env_episode
            )

        if done:
            day_array[
                thread_id
            ] = 0

            episode_array[
                thread_id
            ] += 1
        else:
            day_array[
                thread_id
            ] += 1

    def _open_csv_files(
        self,
    ):
        reservoir_path = os.path.join(
            self.run_dir,
            "cascade_reservoir_timeseries.csv",
        )

        system_path = os.path.join(
            self.run_dir,
            "cascade_system_timeseries.csv",
        )

        constraint_path = os.path.join(
            self.run_dir,
            "cascade_constraint_timeseries.csv",
        )

        lagrangian_path = os.path.join(
            self.run_dir,
            "cascade_lagrangian_updates.csv",
        )

        self._reservoir_file = open(
            reservoir_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self._system_file = open(
            system_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self._constraint_file = open(
            constraint_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self._lagrangian_file = open(
            lagrangian_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self._reservoir_writer = (
            csv.DictWriter(
                self._reservoir_file,
                fieldnames=[
                    "phase",
                    "training_step",
                    "phase_step",
                    "update",
                    "rollout_step",
                    "thread",
                    "env_episode",
                    "date",
                    "reservoir_id",
                    "inflow_m3s",
                    "release_m3s",
                    "next_storage_m3",
                    "next_level_m",
                    "power_mw",
                    "energy_mwh",
                    "safety_mode",
                    "p4_release_inflow_ratio",
                    "p3_relaxation_fraction",
                    "p2_total_violation",
                ],
            )
        )

        self._system_writer = (
            csv.DictWriter(
                self._system_file,
                fieldnames=[
                    "phase",
                    "training_step",
                    "phase_step",
                    "update",
                    "rollout_step",
                    "thread",
                    "env_episode",
                    "date",
                    "reward",
                    "system_energy_mwh",
                    "total_power_mw",
                    "total_storage_m3",
                    "total_release_m3s",
                    "safety_mode",
                    "p4_release_inflow_ratio",
                    "p3_relaxation_fraction",
                    "p2_total_violation",
                    "done",
                ],
            )
        )

        self._constraint_writer = (
            csv.DictWriter(
                self._constraint_file,
                fieldnames=[
                    "phase",
                    "training_step",
                    "phase_step",
                    "update",
                    "rollout_step",
                    "thread",
                    "env_episode",
                    "date",
                    "constraint_id",
                    "raw_violation",
                    "normalized_violation",
                    "cost",
                    "active",
                    "budget",
                ],
            )
        )

        self._lagrangian_writer = (
            csv.DictWriter(
                self._lagrangian_file,
                fieldnames=[
                    "update",
                    "training_step",
                    "constraint_id",
                    "lambda",
                    "mean_cost",
                    "budget",
                    "budget_gap",
                    "active_in_batch",
                ],
            )
        )

        self._reservoir_writer.writeheader()
        self._system_writer.writeheader()
        self._constraint_writer.writeheader()
        self._lagrangian_writer.writeheader()

        self._flush_files()

    def _flush_files(
        self,
    ):
        for file_object in (
            getattr(
                self,
                "_reservoir_file",
                None,
            ),
            getattr(
                self,
                "_system_file",
                None,
            ),
            getattr(
                self,
                "_constraint_file",
                None,
            ),
            getattr(
                self,
                "_lagrangian_file",
                None,
            ),
        ):
            if file_object is not None:
                file_object.flush()

        self._steps_since_flush = 0

    def _constraint_vector(
        self,
        values,
        name,
    ):
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        expected_shape = (
            len(
                self.constraint_ids
            ),
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, "
                f"received {array.shape}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                f"{name} contains "
                "non-finite values"
            )

        return array

    def _optional_constraint_vector(
        self,
        values,
    ):
        if values is None:
            return np.full(
                len(
                    self.constraint_ids
                ),
                np.nan,
                dtype=np.float64,
            )

        return self._constraint_vector(
            values,
            "constraint violation vector",
        )

    @staticmethod
    def _required_number(
        info,
        keys: Sequence[str],
        name,
    ):
        for key in keys:
            if key in info:
                value = float(
                    info[
                        key
                    ]
                )

                if not np.isfinite(
                    value
                ):
                    raise ValueError(
                        f"{name} must be finite"
                    )

                return value

        raise KeyError(
            f"environment info missing {name}; "
            f"expected one of {tuple(keys)}"
        )

    @staticmethod
    def _optional_number(
        info,
        key,
    ):
        if key not in info:
            return float(
                "nan"
            )

        value = float(
            info[
                key
            ]
        )

        if not np.isfinite(
            value
        ):
            return float(
                "nan"
            )

        return value

    @staticmethod
    def _append_metric(
        store,
        key,
        value,
    ):
        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return

        if key not in store:
            store[
                key
            ] = []

        store[
            key
        ].append(
            value
        )