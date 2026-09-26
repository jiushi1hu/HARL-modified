from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Sequence, Tuple

import numpy as np

from harl.common.buffers.on_policy_critic_buffer_ep import (
    OnPolicyCriticBufferEP,
)


@dataclass(frozen=True)
class ConstraintBatchStatistics:
    constraint_ids: Tuple[str, ...]
    discounted_active_means: np.ndarray
    active_weight_sums: np.ndarray
    active_counts: np.ndarray
    valid_flags: np.ndarray
    budgets: np.ndarray

    @property
    def budget_gaps(self) -> np.ndarray:
        return (
            self.discounted_active_means
            - self.budgets
        )


class ConstraintBuffer:

    def __init__(
        self,
        args,
        share_obs_space,
        constraint_ids: Sequence[str],
        budgets: Sequence[float],
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
            self._validate_budgets(
                budgets,
                self.num_constraints,
            )
        )

        self._buffers = tuple(
            OnPolicyCriticBufferEP(
                args,
                share_obs_space,
            )
            for _ in range(
                self.num_constraints
            )
        )

        reference_buffer = (
            self._buffers[0]
        )

        self.episode_length = (
            reference_buffer.episode_length
        )

        self.n_rollout_threads = (
            reference_buffer
            .n_rollout_threads
        )

        self.recurrent_n = (
            reference_buffer.recurrent_n
        )

        self.rnn_hidden_size = (
            reference_buffer
            .rnn_hidden_size
        )

        self.gamma = float(
            reference_buffer.gamma
        )

        self.gae_lambda = float(
            reference_buffer.gae_lambda
        )

        self.active_flags = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                self.num_constraints,
            ),
            dtype=np.float32,
        )

    @property
    def step(self) -> int:
        steps = {
            buffer.step
            for buffer in self._buffers
        }

        if len(steps) != 1:
            raise RuntimeError(
                "constraint buffers are "
                "out of sync"
            )

        return int(
            next(iter(steps))
        )

    @property
    def costs(self) -> np.ndarray:
        return np.stack(
            [
                buffer.rewards[
                    :,
                    :,
                    0,
                ]
                for buffer in self._buffers
            ],
            axis=-1,
        )

    @property
    def value_preds(self) -> np.ndarray:
        return np.stack(
            [
                buffer.value_preds[
                    :,
                    :,
                    0,
                ]
                for buffer in self._buffers
            ],
            axis=-1,
        )

    @property
    def returns(self) -> np.ndarray:
        return np.stack(
            [
                buffer.returns[
                    :,
                    :,
                    0,
                ]
                for buffer in self._buffers
            ],
            axis=-1,
        )

    def initialize(
        self,
        share_obs,
    ) -> None:
        share_obs = np.asarray(
            share_obs,
            dtype=np.float32,
        )

        expected_shape = (
            self._buffers[0]
            .share_obs[0]
            .shape
        )

        if (
            share_obs.shape
            != expected_shape
        ):
            raise ValueError(
                "initial share_obs must have "
                f"shape {expected_shape}, "
                f"received {share_obs.shape}"
            )

        if not np.all(
            np.isfinite(
                share_obs
            )
        ):
            raise ValueError(
                "initial share_obs contains "
                "non-finite values"
            )

        for buffer in self._buffers:
            buffer.share_obs[
                0
            ] = share_obs.copy()

    def insert(
        self,
        share_obs,
        rnn_states_critic,
        value_preds,
        costs,
        active_flags,
        masks,
        bad_masks,
    ) -> None:

        current_step = self.step

        share_obs = self._prepare_share_obs(
            share_obs
        )

        rnn_states_critic = (
            self._prepare_rnn_states(
                rnn_states_critic
            )
        )

        value_preds = (
            self._prepare_constraint_matrix(
                value_preds,
                "value_preds",
            )
        )

        costs = (
            self._prepare_constraint_matrix(
                costs,
                "costs",
            )
        )

        if np.any(
            costs < 0.0
        ):
            raise ValueError(
                "constraint costs cannot "
                "be negative"
            )

        active_flags = (
            self._prepare_active_flags(
                active_flags
            )
        )

        inactive_costs = costs[
            active_flags == 0.0
        ]

        if (
            inactive_costs.size > 0
            and np.any(
                np.abs(
                    inactive_costs
                )
                > 1e-8
            )
        ):
            raise ValueError(
                "inactive constraints must "
                "have zero cost"
            )

        masks = self._prepare_masks(
            masks,
            "masks",
        )

        bad_masks = self._prepare_masks(
            bad_masks,
            "bad_masks",
        )

        self.active_flags[
            current_step
        ] = active_flags

        for constraint_index, buffer in enumerate(
            self._buffers
        ):
            buffer.insert(
                share_obs=share_obs,
                rnn_states_critic=(
                    rnn_states_critic[
                        :,
                        constraint_index,
                    ]
                ),
                value_preds=(
                    value_preds[
                        :,
                        constraint_index:
                        constraint_index + 1,
                    ]
                ),
                rewards=(
                    costs[
                        :,
                        constraint_index:
                        constraint_index + 1,
                    ]
                ),
                masks=masks,
                bad_masks=bad_masks,
            )

        self._validate_synchronization()

    def compute_returns(
        self,
        next_values,
        value_normalizers=None,
    ) -> None:

        next_values = (
            self._prepare_constraint_matrix(
                next_values,
                "next_values",
            )
        )

        normalizers = (
            self._prepare_value_normalizers(
                value_normalizers
            )
        )

        for constraint_index, buffer in enumerate(
            self._buffers
        ):
            buffer.compute_returns(
                next_value=(
                    next_values[
                        :,
                        constraint_index:
                        constraint_index + 1,
                    ]
                ),
                value_normalizer=(
                    normalizers[
                        constraint_index
                    ]
                ),
            )

    def compute_advantages(
        self,
        value_normalizers=None,
    ) -> np.ndarray:

        normalizers = (
            self._prepare_value_normalizers(
                value_normalizers
            )
        )

        advantages = np.empty(
            (
                self.episode_length,
                self.n_rollout_threads,
                self.num_constraints,
            ),
            dtype=np.float32,
        )

        for constraint_index, (
            buffer,
            normalizer,
        ) in enumerate(
            zip(
                self._buffers,
                normalizers,
            )
        ):
            value_preds = (
                buffer.value_preds[
                    :-1
                ]
            )

            if normalizer is not None:
                value_preds = (
                    self._to_numpy(
                        normalizer.denormalize(
                            value_preds
                        )
                    )
                )

            advantages[
                :,
                :,
                constraint_index,
            ] = (
                buffer.returns[
                    :-1,
                    :,
                    0,
                ]
                - value_preds[
                    :,
                    :,
                    0,
                ]
            )

        return advantages

    def compute_discounted_active_statistics(
        self,
        gamma: Optional[float] = None,
    ) -> ConstraintBatchStatistics:

        if gamma is None:
            gamma = self.gamma

        gamma = float(
            gamma
        )

        if (
            not np.isfinite(
                gamma
            )
            or gamma <= 0.0
            or gamma > 1.0
        ):
            raise ValueError(
                "gamma must be in (0, 1]"
            )

        costs = self.costs.astype(
            np.float64,
            copy=False,
        )

        active_flags = (
            self.active_flags.astype(
                np.float64,
                copy=False,
            )
        )

        time_weights = np.power(
            gamma,
            np.arange(
                self.episode_length,
                dtype=np.float64,
            ),
        )

        weighted_active = (
            time_weights[
                :,
                np.newaxis,
                np.newaxis,
            ]
            * active_flags
        )

        active_weight_sums = np.sum(
            weighted_active,
            axis=(0, 1),
        )

        weighted_cost_sums = np.sum(
            weighted_active
            * costs,
            axis=(0, 1),
        )

        active_counts = np.sum(
            active_flags,
            axis=(0, 1),
        ).astype(
            np.int64
        )

        valid_flags = (
            active_weight_sums
            > 0.0
        )

        means = np.full(
            self.num_constraints,
            np.nan,
            dtype=np.float64,
        )

        np.divide(
            weighted_cost_sums,
            active_weight_sums,
            out=means,
            where=valid_flags,
        )

        means = means.astype(
            np.float32
        )

        active_weight_sums = (
            active_weight_sums.astype(
                np.float32
            )
        )

        return ConstraintBatchStatistics(
            constraint_ids=(
                self.constraint_ids
            ),
            discounted_active_means=(
                self._readonly_copy(
                    means
                )
            ),
            active_weight_sums=(
                self._readonly_copy(
                    active_weight_sums
                )
            ),
            active_counts=(
                self._readonly_copy(
                    active_counts
                )
            ),
            valid_flags=(
                self._readonly_copy(
                    valid_flags
                )
            ),
            budgets=self.budgets,
        )

    def get_buffer(
        self,
        constraint,
    ) -> OnPolicyCriticBufferEP:

        index = self._get_constraint_index(
            constraint
        )

        return self._buffers[
            index
        ]

    def get_constraint_index(
        self,
        constraint_id: str,
    ) -> int:
        return self._get_constraint_index(
            constraint_id
        )

    def after_update(self) -> None:

        for buffer in self._buffers:
            buffer.after_update()

        self.active_flags.fill(
            0.0
        )

        self._validate_synchronization()

    def get_mean_active_costs(
        self,
    ) -> np.ndarray:

        costs = self.costs
        flags = self.active_flags

        means = np.full(
            self.num_constraints,
            np.nan,
            dtype=np.float32,
        )

        for index in range(
            self.num_constraints
        ):
            active = (
                flags[
                    :,
                    :,
                    index
                ]
                > 0.0
            )

            if np.any(
                active
            ):
                means[
                    index
                ] = float(
                    np.mean(
                        costs[
                            :,
                            :,
                            index
                        ][active]
                    )
                )

        return means

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
                    "constraint index out "
                    "of range"
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

    def _prepare_share_obs(
        self,
        values,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        expected_shape = (
            self._buffers[0]
            .share_obs[
                self.step + 1
            ]
            .shape
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                "share_obs must have "
                f"shape {expected_shape}, "
                f"received {array.shape}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "share_obs contains "
                "non-finite values"
            )

        return array

    def _prepare_rnn_states(
        self,
        values,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        expected_shape = (
            self.n_rollout_threads,
            self.num_constraints,
            self.recurrent_n,
            self.rnn_hidden_size,
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                "rnn_states_critic must "
                f"have shape {expected_shape}, "
                f"received {array.shape}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "rnn_states_critic contains "
                "non-finite values"
            )

        return array

    def _prepare_constraint_matrix(
        self,
        values,
        name: str,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if (
            array.ndim == 3
            and array.shape[-1] == 1
        ):
            array = array[
                :,
                :,
                0,
            ]

        expected_shape = (
            self.n_rollout_threads,
            self.num_constraints,
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                f"{name} must have "
                f"shape {expected_shape}, "
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

    def _prepare_active_flags(
        self,
        values,
    ) -> np.ndarray:

        array = (
            self._prepare_constraint_matrix(
                values,
                "active_flags",
            )
        )

        if not np.all(
            (array == 0.0)
            | (array == 1.0)
        ):
            raise ValueError(
                "active_flags must contain "
                "only 0 or 1"
            )

        return array

    def _prepare_masks(
        self,
        values,
        name: str,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        expected_shape = (
            self.n_rollout_threads,
            1,
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                f"{name} must have "
                f"shape {expected_shape}, "
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

    def _prepare_value_normalizers(
        self,
        value_normalizers,
    ):

        if value_normalizers is None:
            return tuple(
                None
                for _ in range(
                    self.num_constraints
                )
            )

        value_normalizers = tuple(
            value_normalizers
        )

        if (
            len(
                value_normalizers
            )
            != self.num_constraints
        ):
            raise ValueError(
                "value_normalizers length "
                "must equal num_constraints"
            )

        return value_normalizers

    def _validate_synchronization(
        self,
    ) -> None:

        steps = [
            buffer.step
            for buffer in self._buffers
        ]

        if len(
            set(
                steps
            )
        ) != 1:
            raise RuntimeError(
                "constraint buffers became "
                "out of sync"
            )

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
    def _validate_budgets(
        budgets,
        num_constraints,
    ) -> np.ndarray:

        array = np.asarray(
            budgets,
            dtype=np.float32,
        )

        if array.shape != (
            num_constraints,
        ):
            raise ValueError(
                "budgets must have shape "
                f"({num_constraints},)"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "budgets contain "
                "non-finite values"
            )

        if np.any(
            array < 0.0
        ):
            raise ValueError(
                "budgets cannot be negative"
            )

        array = array.copy()

        array.setflags(
            write=False
        )

        return array

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

    @staticmethod
    def _to_numpy(
        value,
    ) -> np.ndarray:

        if hasattr(
            value,
            "detach",
        ):
            return (
                value.detach()
                .cpu()
                .numpy()
            )

        return np.asarray(
            value
        )