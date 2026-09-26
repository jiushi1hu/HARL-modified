from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from harl.common.buffers.constraint_buffer import (
    ConstraintBatchStatistics,
)


_STATE_FILE_NAME = "lagrangian_state.npz"


@dataclass(frozen=True)
class LagrangianUpdateResult:
    constraint_ids: Tuple[str, ...]
    old_multipliers: np.ndarray
    new_multipliers: np.ndarray
    mean_costs: np.ndarray
    budgets: np.ndarray
    budget_gaps: np.ndarray
    valid_flags: np.ndarray

    @property
    def updated_flags(self) -> np.ndarray:
        return self.valid_flags.copy()


class LagrangianManager:

    def __init__(
        self,
        constraint_ids: Sequence[str],
        budgets,
        learning_rates,
        initial_values=0.0,
    ):
        self.constraint_ids = (
            self._validate_constraint_ids(
                constraint_ids
            )
        )

        self.num_constraints = len(
            self.constraint_ids
        )

        self.budgets = (
            self._prepare_vector(
                budgets,
                name="budgets",
                allow_zero=True,
            )
        )

        self.learning_rates = (
            self._prepare_vector(
                learning_rates,
                name="learning_rates",
                allow_zero=False,
            )
        )

        self._multipliers = (
            self._prepare_vector(
                initial_values,
                name="initial_values",
                allow_zero=True,
            )
        )

    @classmethod
    def from_config(
        cls,
        env_args: Mapping,
        constraint_ids: Sequence[str],
        budgets,
    ) -> "LagrangianManager":

        lagrangian_config = env_args.get(
            "lagrangian",
            {},
        )

        default_initial_value = float(
            lagrangian_config.get(
                "initial_value",
                0.0,
            )
        )

        default_learning_rate = float(
            lagrangian_config.get(
                "learning_rate",
                0.05,
            )
        )

        constraint_config = env_args.get(
            "long_term_constraints",
            ()
        )

        config_by_id = {}

        for item in constraint_config:
            if not isinstance(
                item,
                Mapping,
            ):
                raise TypeError(
                    "long_term_constraints entries "
                    "must be mappings"
                )

            reservoir_id = str(
                item["reservoir_id"]
            )

            constraint_type = str(
                item["type"]
            )

            constraint_id = (
                f"{reservoir_id}:"
                f"{constraint_type}"
            )

            if constraint_id in config_by_id:
                raise ValueError(
                    "duplicate long-term "
                    f"constraint: {constraint_id}"
                )

            config_by_id[
                constraint_id
            ] = item

        initial_values = []
        learning_rates = []

        for constraint_id in (
            constraint_ids
        ):
            item = config_by_id.get(
                constraint_id,
                {},
            )

            initial_values.append(
                float(
                    item.get(
                        "lambda_init",
                        default_initial_value,
                    )
                )
            )

            learning_rates.append(
                float(
                    item.get(
                        "lambda_lr",
                        default_learning_rate,
                    )
                )
            )

        return cls(
            constraint_ids=constraint_ids,
            budgets=budgets,
            learning_rates=(
                learning_rates
            ),
            initial_values=(
                initial_values
            ),
        )

    @property
    def multipliers(self) -> np.ndarray:
        return self._readonly_copy(
            self._multipliers
        )

    def snapshot(self) -> np.ndarray:
        return self._readonly_copy(
            self._multipliers
        )

    def get_multiplier(
        self,
        constraint,
    ) -> float:

        index = (
            self._get_constraint_index(
                constraint
            )
        )

        return float(
            self._multipliers[
                index
            ]
        )

    def combine_advantages(
        self,
        reward_advantages,
        constraint_advantages,
        multipliers=None,
    ) -> np.ndarray:

        reward_advantages = np.asarray(
            reward_advantages,
            dtype=np.float32,
        )

        constraint_advantages = np.asarray(
            constraint_advantages,
            dtype=np.float32,
        )

        if reward_advantages.ndim < 2:
            raise ValueError(
                "reward_advantages must include "
                "time and rollout dimensions"
            )

        if reward_advantages.shape[-1] != 1:
            raise ValueError(
                "reward_advantages last dimension "
                "must be 1"
            )

        expected_constraint_shape = (
            reward_advantages.shape[:-1]
            + (
                self.num_constraints,
            )
        )

        if (
            constraint_advantages.shape
            != expected_constraint_shape
        ):
            raise ValueError(
                "constraint_advantages must have "
                f"shape {expected_constraint_shape}, "
                f"received "
                f"{constraint_advantages.shape}"
            )

        if not np.all(
            np.isfinite(
                reward_advantages
            )
        ):
            raise ValueError(
                "reward_advantages contains "
                "non-finite values"
            )

        if not np.all(
            np.isfinite(
                constraint_advantages
            )
        ):
            raise ValueError(
                "constraint_advantages contains "
                "non-finite values"
            )

        if multipliers is None:
            multiplier_values = (
                self._multipliers
            )
        else:
            multiplier_values = (
                self._prepare_vector(
                    multipliers,
                    name="multipliers",
                    allow_zero=True,
                )
            )

        multiplier_shape = (
            (1,)
            * (
                constraint_advantages.ndim
                - 1
            )
            + (
                self.num_constraints,
            )
        )

        weighted_constraint_advantage = (
            constraint_advantages
            * multiplier_values.reshape(
                multiplier_shape
            )
        )

        total_constraint_penalty = (
            np.sum(
                weighted_constraint_advantage,
                axis=-1,
                keepdims=True,
            )
        )

        lagrangian_advantage = (
            reward_advantages
            - total_constraint_penalty
        )

        return lagrangian_advantage.astype(
            np.float32,
            copy=False,
        )

    def update(
        self,
        statistics: ConstraintBatchStatistics,
    ) -> LagrangianUpdateResult:

        if not isinstance(
            statistics,
            ConstraintBatchStatistics,
        ):
            raise TypeError(
                "statistics must be "
                "ConstraintBatchStatistics"
            )

        if (
            statistics.constraint_ids
            != self.constraint_ids
        ):
            raise ValueError(
                "constraint IDs in statistics "
                "do not match LagrangianManager"
            )

        statistic_budgets = np.asarray(
            statistics.budgets,
            dtype=np.float32,
        )

        if not np.allclose(
            statistic_budgets,
            self.budgets,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "constraint budgets in statistics "
                "do not match LagrangianManager"
            )

        mean_costs = np.asarray(
            statistics
            .discounted_active_means,
            dtype=np.float32,
        )

        valid_flags = np.asarray(
            statistics.valid_flags,
            dtype=bool,
        )

        expected_shape = (
            self.num_constraints,
        )

        if mean_costs.shape != expected_shape:
            raise ValueError(
                "mean_costs has invalid shape"
            )

        if valid_flags.shape != expected_shape:
            raise ValueError(
                "valid_flags has invalid shape"
            )

        if np.any(
            valid_flags
            & ~np.isfinite(
                mean_costs
            )
        ):
            raise ValueError(
                "active constraints must have "
                "finite mean costs"
            )

        old_multipliers = (
            self._multipliers.copy()
        )

        budget_gaps = (
            mean_costs
            - self.budgets
        )

        for constraint_index in range(
            self.num_constraints
        ):
            if not valid_flags[
                constraint_index
            ]:
                continue

            updated_value = (
                self._multipliers[
                    constraint_index
                ]
                + self.learning_rates[
                    constraint_index
                ]
                * budget_gaps[
                    constraint_index
                ]
            )

            self._multipliers[
                constraint_index
            ] = max(
                0.0,
                float(
                    updated_value
                ),
            )

        return LagrangianUpdateResult(
            constraint_ids=(
                self.constraint_ids
            ),
            old_multipliers=(
                self._readonly_copy(
                    old_multipliers
                )
            ),
            new_multipliers=(
                self._readonly_copy(
                    self._multipliers
                )
            ),
            mean_costs=(
                self._readonly_copy(
                    mean_costs
                )
            ),
            budgets=(
                self.budgets
            ),
            budget_gaps=(
                self._readonly_copy(
                    budget_gaps
                )
            ),
            valid_flags=(
                self._readonly_copy(
                    valid_flags
                )
            ),
        )

    def state_dict(self) -> dict:
        return {
            "constraint_ids": (
                self.constraint_ids
            ),
            "budgets": (
                self.budgets.copy()
            ),
            "learning_rates": (
                self.learning_rates.copy()
            ),
            "multipliers": (
                self._multipliers.copy()
            ),
        }

    def load_state_dict(
        self,
        state_dict,
    ) -> None:

        if not isinstance(
            state_dict,
            Mapping,
        ):
            raise TypeError(
                "state_dict must be a mapping"
            )

        constraint_ids = tuple(
            state_dict[
                "constraint_ids"
            ]
        )

        if constraint_ids != self.constraint_ids:
            raise ValueError(
                "state_dict constraint IDs "
                "do not match current "
                "configuration"
            )

        budgets = np.asarray(
            state_dict[
                "budgets"
            ],
            dtype=np.float32,
        )

        if not np.allclose(
            budgets,
            self.budgets,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "state_dict budgets do not "
                "match current configuration"
            )

        learning_rates = np.asarray(
            state_dict[
                "learning_rates"
            ],
            dtype=np.float32,
        )

        if not np.allclose(
            learning_rates,
            self.learning_rates,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "state_dict learning rates "
                "do not match current "
                "configuration"
            )

        multipliers = (
            self._prepare_vector(
                state_dict[
                    "multipliers"
                ],
                name="multipliers",
                allow_zero=True,
            )
        )

        self._multipliers[:] = (
            multipliers
        )

    def save(
        self,
        save_dir,
    ) -> None:

        save_dir = Path(
            save_dir
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            save_dir
            / _STATE_FILE_NAME,
            constraint_ids=np.asarray(
                self.constraint_ids,
                dtype=str,
            ),
            budgets=self.budgets,
            learning_rates=(
                self.learning_rates
            ),
            multipliers=(
                self._multipliers
            ),
        )

    def restore(
        self,
        model_dir,
    ) -> None:

        path = (
            Path(
                model_dir
            )
            / _STATE_FILE_NAME
        )

        if not path.exists():
            raise FileNotFoundError(
                "Lagrangian state not found: "
                f"{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as state:

            constraint_ids = tuple(
                str(value)
                for value
                in state[
                    "constraint_ids"
                ].tolist()
            )

            self.load_state_dict(
                {
                    "constraint_ids": (
                        constraint_ids
                    ),
                    "budgets": state[
                        "budgets"
                    ],
                    "learning_rates": state[
                        "learning_rates"
                    ],
                    "multipliers": state[
                        "multipliers"
                    ],
                }
            )

    def _get_constraint_index(
        self,
        constraint,
    ) -> int:

        if (
            isinstance(
                constraint,
                Integral,
            )
            and not isinstance(
                constraint,
                (bool, np.bool_),
            )
        ):
            index = int(
                constraint
            )

            if not (
                0
                <= index
                < self.num_constraints
            ):
                raise IndexError(
                    "constraint index "
                    "out of range"
                )

            return index

        constraint_id = str(
            constraint
        )

        try:
            return (
                self.constraint_ids.index(
                    constraint_id
                )
            )
        except ValueError as exc:
            raise KeyError(
                f"unknown constraint_id: "
                f"{constraint_id}"
            ) from exc

    def _prepare_vector(
        self,
        values,
        name: str,
        allow_zero: bool,
    ) -> np.ndarray:

        if np.isscalar(
            values
        ):
            array = np.full(
                self.num_constraints,
                float(values),
                dtype=np.float32,
            )
        else:
            array = np.asarray(
                values,
                dtype=np.float32,
            )

        expected_shape = (
            self.num_constraints,
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

        if allow_zero:
            invalid = (
                array < 0.0
            )
        else:
            invalid = (
                array <= 0.0
            )

        if np.any(
            invalid
        ):
            relation = (
                "nonnegative"
                if allow_zero
                else "positive"
            )

            raise ValueError(
                f"{name} must be "
                f"{relation}"
            )

        return array.copy()

    @staticmethod
    def _validate_constraint_ids(
        constraint_ids,
    ) -> Tuple[str, ...]:

        if isinstance(
            constraint_ids,
            (str, bytes),
        ):
            raise TypeError(
                "constraint_ids must be "
                "a sequence"
            )

        constraint_ids = tuple(
            str(value)
            for value
            in constraint_ids
        )

        if len(
            constraint_ids
        ) == 0:
            raise ValueError(
                "constraint_ids cannot "
                "be empty"
            )

        if any(
            not value
            for value
            in constraint_ids
        ):
            raise ValueError(
                "constraint_id cannot "
                "be empty"
            )

        if (
            len(
                set(
                    constraint_ids
                )
            )
            != len(
                constraint_ids
            )
        ):
            raise ValueError(
                "constraint_ids must "
                "be unique"
            )

        return constraint_ids

    @staticmethod
    def _readonly_copy(
        values,
    ) -> np.ndarray:

        array = np.asarray(
            values
        ).copy()

        array.setflags(
            write=False
        )

        return array
