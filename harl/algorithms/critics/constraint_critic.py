from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from harl.algorithms.critics.v_critic import VCritic
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.constraint_buffer import (
    ConstraintBuffer,
)


_CHECKPOINT_NAME = "constraint_critics.pt"


class ConstraintCriticSet:

    def __init__(
        self,
        critic_args,
        share_obs_space,
        constraint_ids: Sequence[str],
        device=torch.device("cpu"),
        use_valuenorm: bool = True,
    ):
        self.constraint_ids = (
            self._validate_constraint_ids(
                constraint_ids
            )
        )

        self.num_constraints = len(
            self.constraint_ids
        )

        self.device = device

        self.use_valuenorm = bool(
            use_valuenorm
        )

        self.recurrent_n = int(
            critic_args["recurrent_n"]
        )

        hidden_sizes = tuple(
            critic_args["hidden_sizes"]
        )

        if len(hidden_sizes) == 0:
            raise ValueError(
                "hidden_sizes cannot be empty"
            )

        self.rnn_hidden_size = int(
            hidden_sizes[-1]
        )

        self.critics = tuple(
            VCritic(
                critic_args,
                share_obs_space,
                device=self.device,
            )
            for _ in range(
                self.num_constraints
            )
        )

        if self.use_valuenorm:
            self.value_normalizers = tuple(
                ValueNorm(
                    1,
                    device=self.device,
                )
                for _ in range(
                    self.num_constraints
                )
            )
        else:
            self.value_normalizers = tuple(
                None
                for _ in range(
                    self.num_constraints
                )
            )

    def get_values(
        self,
        share_obs,
        rnn_states_critic,
        masks,
    ):
        share_obs = self._prepare_share_obs(
            share_obs
        )

        batch_size = (
            share_obs.shape[0]
        )

        rnn_states_critic = (
            self._prepare_rnn_states(
                rnn_states_critic,
                batch_size,
            )
        )

        masks = self._prepare_masks(
            masks,
            batch_size,
        )

        value_list = []
        rnn_state_list = []

        for constraint_index, critic in enumerate(
            self.critics
        ):
            value, next_rnn_state = (
                critic.get_values(
                    share_obs,
                    rnn_states_critic[
                        :,
                        constraint_index,
                    ],
                    masks,
                )
            )

            value_list.append(
                value
            )

            rnn_state_list.append(
                next_rnn_state
            )

        values = torch.stack(
            value_list,
            dim=1,
        )

        next_rnn_states = torch.stack(
            rnn_state_list,
            dim=1,
        )

        return (
            values,
            next_rnn_states,
        )

    @torch.no_grad()
    def compute_returns(
        self,
        constraint_buffer: ConstraintBuffer,
    ) -> None:

        self._validate_buffer(
            constraint_buffer
        )

        next_values = np.empty(
            (
                constraint_buffer
                .n_rollout_threads,
                self.num_constraints,
            ),
            dtype=np.float32,
        )

        for constraint_index, critic in enumerate(
            self.critics
        ):
            buffer = (
                constraint_buffer
                .get_buffer(
                    constraint_index
                )
            )

            next_value, _ = (
                critic.get_values(
                    buffer.share_obs[-1],
                    buffer.rnn_states_critic[-1],
                    buffer.masks[-1],
                )
            )

            next_values[
                :,
                constraint_index,
            ] = (
                self._to_numpy(
                    next_value
                )[
                    :,
                    0,
                ]
            )

        constraint_buffer.compute_returns(
            next_values=next_values,
            value_normalizers=(
                self.value_normalizers
            ),
        )

    def compute_advantages(
        self,
        constraint_buffer: ConstraintBuffer,
    ) -> np.ndarray:

        self._validate_buffer(
            constraint_buffer
        )

        return (
            constraint_buffer
            .compute_advantages(
                value_normalizers=(
                    self.value_normalizers
                )
            )
        )

    def train(
        self,
        constraint_buffer: ConstraintBuffer,
    ):
        self._validate_buffer(
            constraint_buffer
        )

        train_info = {}

        for constraint_index, (
            constraint_id,
            critic,
            value_normalizer,
        ) in enumerate(
            zip(
                self.constraint_ids,
                self.critics,
                self.value_normalizers,
            )
        ):
            buffer = (
                constraint_buffer
                .get_buffer(
                    constraint_index
                )
            )

            critic_info = critic.train(
                buffer,
                value_normalizer=(
                    value_normalizer
                ),
            )

            for key, value in (
                critic_info.items()
            ):
                train_info[
                    f"{constraint_id}/{key}"
                ] = self._to_scalar(
                    value
                )

        return train_info

    def prep_rollout(self) -> None:
        for critic in self.critics:
            critic.prep_rollout()

    def prep_training(self) -> None:
        for critic in self.critics:
            critic.prep_training()

    def lr_decay(
        self,
        episode: int,
        episodes: int,
    ) -> None:
        episode = int(
            episode
        )

        episodes = int(
            episodes
        )

        if episode < 0:
            raise ValueError(
                "episode cannot be negative"
            )

        if episodes <= 0:
            raise ValueError(
                "episodes must be positive"
            )

        for critic in self.critics:
            critic.lr_decay(
                episode,
                episodes,
            )

    def get_critic(
        self,
        constraint,
    ) -> VCritic:

        index = (
            self._get_constraint_index(
                constraint
            )
        )

        return self.critics[
            index
        ]

    def get_value_normalizer(
        self,
        constraint,
    ):
        index = (
            self._get_constraint_index(
                constraint
            )
        )

        return self.value_normalizers[
            index
        ]

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

        checkpoint = {
            "constraint_ids": (
                self.constraint_ids
            ),
            "use_valuenorm": (
                self.use_valuenorm
            ),
            "critic_state_dicts": [
                critic.critic.state_dict()
                for critic
                in self.critics
            ],
            "value_normalizer_state_dicts": [
                (
                    normalizer.state_dict()
                    if normalizer
                    is not None
                    else None
                )
                for normalizer
                in self.value_normalizers
            ],
        }

        torch.save(
            checkpoint,
            save_dir
            / _CHECKPOINT_NAME,
        )

    def restore(
        self,
        model_dir,
    ) -> None:

        checkpoint_path = (
            Path(
                model_dir
            )
            / _CHECKPOINT_NAME
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"constraint critic checkpoint "
                f"not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        checkpoint_ids = tuple(
            checkpoint[
                "constraint_ids"
            ]
        )

        if (
            checkpoint_ids
            != self.constraint_ids
        ):
            raise ValueError(
                "checkpoint constraint IDs "
                "do not match current "
                "configuration"
            )

        checkpoint_use_valuenorm = bool(
            checkpoint[
                "use_valuenorm"
            ]
        )

        if (
            checkpoint_use_valuenorm
            != self.use_valuenorm
        ):
            raise ValueError(
                "checkpoint ValueNorm setting "
                "does not match current "
                "configuration"
            )

        critic_state_dicts = (
            checkpoint[
                "critic_state_dicts"
            ]
        )

        if (
            len(
                critic_state_dicts
            )
            != self.num_constraints
        ):
            raise ValueError(
                "checkpoint critic count "
                "does not match "
                "num_constraints"
            )

        for critic, state_dict in zip(
            self.critics,
            critic_state_dicts,
        ):
            critic.critic.load_state_dict(
                state_dict
            )

        if self.use_valuenorm:
            normalizer_state_dicts = (
                checkpoint[
                    "value_normalizer_state_dicts"
                ]
            )

            if (
                len(
                    normalizer_state_dicts
                )
                != self.num_constraints
            ):
                raise ValueError(
                    "checkpoint ValueNorm count "
                    "does not match "
                    "num_constraints"
                )

            for (
                normalizer,
                state_dict,
            ) in zip(
                self.value_normalizers,
                normalizer_state_dicts,
            ):
                if state_dict is None:
                    raise ValueError(
                        "missing ValueNorm state"
                    )

                normalizer.load_state_dict(
                    state_dict
                )

    def zero_rnn_states(
        self,
        batch_size: int,
    ) -> np.ndarray:

        batch_size = int(
            batch_size
        )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        return np.zeros(
            (
                batch_size,
                self.num_constraints,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

    def _validate_buffer(
        self,
        constraint_buffer,
    ) -> None:

        if not isinstance(
            constraint_buffer,
            ConstraintBuffer,
        ):
            raise TypeError(
                "constraint_buffer must be "
                "ConstraintBuffer"
            )

        if (
            constraint_buffer
            .constraint_ids
            != self.constraint_ids
        ):
            raise ValueError(
                "constraint critic IDs "
                "do not match "
                "constraint buffer IDs"
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

    def _prepare_share_obs(
        self,
        values,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim < 2:
            raise ValueError(
                "share_obs must include "
                "a batch dimension"
            )

        if array.shape[0] == 0:
            raise ValueError(
                "share_obs batch "
                "cannot be empty"
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
        batch_size: int,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        expected_shape = (
            batch_size,
            self.num_constraints,
            self.recurrent_n,
            self.rnn_hidden_size,
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                "constraint critic RNN states "
                f"must have shape "
                f"{expected_shape}, "
                f"received {array.shape}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "constraint critic RNN states "
                "contain non-finite values"
            )

        return array

    @staticmethod
    def _prepare_masks(
        values,
        batch_size: int,
    ) -> np.ndarray:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        expected_shape = (
            batch_size,
            1,
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                "constraint critic masks "
                f"must have shape "
                f"{expected_shape}, "
                f"received {array.shape}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "constraint critic masks "
                "contain non-finite values"
            )

        return array

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
            str(
                value
            )
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
    def _to_numpy(
        value,
    ) -> np.ndarray:

        if isinstance(
            value,
            torch.Tensor,
        ):
            return (
                value.detach()
                .cpu()
                .numpy()
            )

        return np.asarray(
            value
        )

    @staticmethod
    def _to_scalar(
        value,
    ) -> float:

        if isinstance(
            value,
            torch.Tensor,
        ):
            value = (
                value.detach()
                .cpu()
                .item()
            )

        return float(
            value
        )