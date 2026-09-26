from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch

from harl.envs.cascade_reservoir.action_mapping import (
    ReleaseActionMapper,
)
from harl.envs.cascade_reservoir.routing import (
    compute_downstream_inflow,
)
from harl.envs.cascade_reservoir.s3_relaxation import (
    SafetyRecoveryResult,
)


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_DOWNSTREAM_RESERVOIRS = (
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_NUM_RESERVOIRS = len(
    _RESERVOIR_ORDER
)


@dataclass(frozen=True)
class SequentialSamplingResult:
    """Complete S4 sequential sampling result.

    decision_observations
        Actual observations seen by the five Actors.

        WDD:
            o_dec = o_base

        BHT/XLD/XJB/THR:
            o_dec = (o_base, I_current)

    action_masks
        Actual HARL-facing float32 masks used during sampling.

    actions
        Shape:
            (batch, 5, 1)

    action_log_probs
        Old-policy log probabilities corresponding exactly to
        the stored decision observations, masks and actions.

    inflows_m3s
        Actual S4 decision-time inflows.

    target_releases_m3s
        Release values obtained from the five discrete action
        mappers.

    executed_releases_m3s
        Release values used to update the directly downstream
        decision-time inflow.

        Under the current implementation:

            Q_exec == Q_target

        The environment remains authoritative and recomputes
        physical execution independently inside env.step().

    rnn_states
        Next Actor RNN states after the five actual decisions.
    """

    reservoir_order: Tuple[str, ...]

    decision_observations: Tuple[
        np.ndarray,
        ...,
    ]

    action_masks: Tuple[
        np.ndarray,
        ...,
    ]

    actions: np.ndarray

    action_log_probs: np.ndarray

    inflows_m3s: np.ndarray

    target_releases_m3s: np.ndarray

    executed_releases_m3s: np.ndarray

    rnn_states: Tuple[
        np.ndarray,
        ...,
    ]

    def __post_init__(
        self,
    ):
        reservoir_order = tuple(
            str(
                reservoir_id
            )
            for reservoir_id
            in self.reservoir_order
        )

        if (
            reservoir_order
            != _RESERVOIR_ORDER
        ):
            raise ValueError(
                "reservoir_order must be "
                "WDD, BHT, XLD, XJB, THR"
            )

        actions_raw = np.asarray(
            self.actions
        )

        if (
            actions_raw.ndim != 3
            or actions_raw.shape[
                1:
            ]
            != (
                _NUM_RESERVOIRS,
                1,
            )
        ):
            raise ValueError(
                "actions must have shape "
                "(batch, 5, 1)"
            )

        batch_size = int(
            actions_raw.shape[
                0
            ]
        )

        if batch_size <= 0:
            raise ValueError(
                "sampling result batch "
                "cannot be empty"
            )

        if (
            actions_raw.dtype == np.bool_
            or not np.issubdtype(
                actions_raw.dtype,
                np.integer,
            )
        ):
            raise TypeError(
                "actions must contain "
                "integer indices"
            )

        actions = actions_raw.astype(
            np.int64,
            copy=True,
        )

        action_log_probs = np.asarray(
            self.action_log_probs,
            dtype=np.float32,
        )

        if (
            action_log_probs.shape
            != actions.shape
        ):
            raise ValueError(
                "action_log_probs must "
                "match actions shape"
            )

        if not np.all(
            np.isfinite(
                action_log_probs
            )
        ):
            raise ValueError(
                "action_log_probs contains "
                "non-finite values"
            )

        action_log_probs = (
            action_log_probs.copy()
        )

        expected_flow_shape = (
            batch_size,
            _NUM_RESERVOIRS,
        )

        inflows = np.asarray(
            self.inflows_m3s,
            dtype=np.float64,
        )

        target_releases = np.asarray(
            self.target_releases_m3s,
            dtype=np.float64,
        )

        executed_releases = np.asarray(
            self.executed_releases_m3s,
            dtype=np.float64,
        )

        for name, values in (
            (
                "inflows_m3s",
                inflows,
            ),
            (
                "target_releases_m3s",
                target_releases,
            ),
            (
                "executed_releases_m3s",
                executed_releases,
            ),
        ):
            if (
                values.shape
                != expected_flow_shape
            ):
                raise ValueError(
                    f"{name} must have "
                    f"shape {expected_flow_shape}"
                )

            if not np.all(
                np.isfinite(
                    values
                )
            ):
                raise ValueError(
                    f"{name} contains "
                    "non-finite values"
                )

            if np.any(
                values < 0.0
            ):
                raise ValueError(
                    f"{name} cannot contain "
                    "negative values"
                )

        # Current project invariant:
        #
        #     Q_exec == Q_target
        #
        # If release clipping/projection is introduced in the
        # future, this invariant and the S202/S4 interface must
        # be revised together.
        if not np.array_equal(
            target_releases,
            executed_releases,
        ):
            raise ValueError(
                "current S4 implementation "
                "requires Q_exec == Q_target"
            )

        inflows = inflows.copy()
        target_releases = (
            target_releases.copy()
        )
        executed_releases = (
            executed_releases.copy()
        )

        decision_observations = tuple(
            np.asarray(
                observation,
                dtype=np.float32,
            )
            for observation
            in self.decision_observations
        )

        if (
            len(
                decision_observations
            )
            != _NUM_RESERVOIRS
        ):
            raise ValueError(
                "exactly five decision "
                "observation arrays "
                "are required"
            )

        readonly_observations = []

        for reservoir_id, observation in zip(
            _RESERVOIR_ORDER,
            decision_observations,
        ):
            if observation.ndim != 2:
                raise ValueError(
                    f"{reservoir_id}: "
                    "decision observation "
                    "must be two-dimensional"
                )

            if (
                observation.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "decision observation "
                    "has invalid batch size"
                )

            if (
                observation.shape[1]
                <= 0
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "decision observation "
                    "cannot be empty"
                )

            if not np.all(
                np.isfinite(
                    observation
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "decision observation "
                    "contains non-finite values"
                )

            observation = (
                observation.copy()
            )

            observation.setflags(
                write=False
            )

            readonly_observations.append(
                observation
            )

        action_masks = tuple(
            np.asarray(
                mask,
                dtype=np.float32,
            )
            for mask
            in self.action_masks
        )

        if (
            len(
                action_masks
            )
            != _NUM_RESERVOIRS
        ):
            raise ValueError(
                "exactly five action "
                "mask arrays are required"
            )

        readonly_masks = []

        for reservoir_index, (
            reservoir_id,
            mask,
        ) in enumerate(
            zip(
                _RESERVOIR_ORDER,
                action_masks,
            )
        ):
            if mask.ndim != 2:
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mask must be "
                    "two-dimensional"
                )

            if (
                mask.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mask has invalid "
                    "batch size"
                )

            if (
                mask.shape[1]
                < 2
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mask must contain "
                    "at least two actions"
                )

            if not np.all(
                np.isfinite(
                    mask
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mask contains "
                    "non-finite values"
                )

            if not np.all(
                (
                    mask == 0.0
                )
                | (
                    mask == 1.0
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mask must be binary"
                )

            if not np.all(
                np.any(
                    mask > 0.0,
                    axis=1,
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mask contains "
                    "an empty batch row"
                )

            selected_actions = actions[
                :,
                reservoir_index,
                0,
            ]

            if np.any(
                selected_actions < 0
            ) or np.any(
                selected_actions
                >= mask.shape[1]
            ):
                raise IndexError(
                    f"{reservoir_id}: "
                    "sampled action outside "
                    "action mask dimension"
                )

            allowed = mask[
                np.arange(
                    batch_size
                ),
                selected_actions,
            ]

            if np.any(
                allowed <= 0.0
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "sampling result contains "
                    "a masked action"
                )

            mask = mask.copy()

            mask.setflags(
                write=False
            )

            readonly_masks.append(
                mask
            )

        rnn_states = tuple(
            np.asarray(
                state,
                dtype=np.float32,
            )
            for state
            in self.rnn_states
        )

        if (
            len(
                rnn_states
            )
            != _NUM_RESERVOIRS
        ):
            raise ValueError(
                "exactly five Actor "
                "RNN states are required"
            )

        readonly_rnn_states = []

        expected_rnn_tail = None

        for reservoir_id, state in zip(
            _RESERVOIR_ORDER,
            rnn_states,
        ):
            if state.ndim != 3:
                raise ValueError(
                    f"{reservoir_id}: "
                    "RNN state must have shape "
                    "(batch, recurrent_n, "
                    "hidden_size)"
                )

            if (
                state.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "RNN state has invalid "
                    "batch size"
                )

            if not np.all(
                np.isfinite(
                    state
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "RNN state contains "
                    "non-finite values"
                )

            current_tail = tuple(
                state.shape[
                    1:
                ]
            )

            if expected_rnn_tail is None:
                expected_rnn_tail = (
                    current_tail
                )

            elif (
                current_tail
                != expected_rnn_tail
            ):
                # CascadeReservoirRunner stacks the five states
                # along the agent dimension.
                raise ValueError(
                    "all five Actor RNN states "
                    "must use the same "
                    "recurrent shape"
                )

            state = state.copy()

            state.setflags(
                write=False
            )

            readonly_rnn_states.append(
                state
            )

        actions.setflags(
            write=False
        )

        action_log_probs.setflags(
            write=False
        )

        inflows.setflags(
            write=False
        )

        target_releases.setflags(
            write=False
        )

        executed_releases.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "reservoir_order",
            reservoir_order,
        )

        object.__setattr__(
            self,
            "decision_observations",
            tuple(
                readonly_observations
            ),
        )

        object.__setattr__(
            self,
            "action_masks",
            tuple(
                readonly_masks
            ),
        )

        object.__setattr__(
            self,
            "actions",
            actions,
        )

        object.__setattr__(
            self,
            "action_log_probs",
            action_log_probs,
        )

        object.__setattr__(
            self,
            "inflows_m3s",
            inflows,
        )

        object.__setattr__(
            self,
            "target_releases_m3s",
            target_releases,
        )

        object.__setattr__(
            self,
            "executed_releases_m3s",
            executed_releases,
        )

        object.__setattr__(
            self,
            "rnn_states",
            tuple(
                readonly_rnn_states
            ),
        )

    @property
    def batch_size(
        self,
    ) -> int:
        return int(
            self.actions.shape[
                0
            ]
        )

    def get_decision_observation(
        self,
        reservoir_id: str,
    ) -> np.ndarray:
        index = self._get_index(
            reservoir_id
        )

        return (
            self.decision_observations[
                index
            ].copy()
        )

    def get_action_mask(
        self,
        reservoir_id: str,
    ) -> np.ndarray:
        index = self._get_index(
            reservoir_id
        )

        return (
            self.action_masks[
                index
            ].copy()
        )

    def get_rnn_state(
        self,
        reservoir_id: str,
    ) -> np.ndarray:
        index = self._get_index(
            reservoir_id
        )

        return (
            self.rnn_states[
                index
            ].copy()
        )

    def _get_index(
        self,
        reservoir_id: str,
    ) -> int:
        reservoir_id = str(
            reservoir_id
        )

        try:
            return (
                self.reservoir_order.index(
                    reservoir_id
                )
            )

        except ValueError as exc:
            raise KeyError(
                "unknown reservoir_id: "
                f"{reservoir_id}"
            ) from exc


class SequentialSampler:
    """S4 upstream-to-downstream behavior-policy sampler.

    The sampler does not advance reservoir physics.

    It only performs the sequential behavior decision:

        WDD
          -> BHT
          -> XLD
          -> XJB
          -> THR

    For every downstream reservoir:

        1. the directly upstream action has already been sampled;
        2. the upstream target/executed release is known;
        3. the current downstream inflow is updated;
        4. the downstream decision observation is formed;
        5. the S203/S3 mask conditioned on the selected upstream
           action is obtained;
        6. the downstream Actor is evaluated and sampled.

    The environment later independently executes the resulting
    five-reservoir joint action.
    """

    def __init__(
        self,
        actors: Mapping[
            str,
            Any,
        ],
        action_mappers: Mapping[
            str,
            ReleaseActionMapper,
        ],
    ):
        self._validate_keys(
            actors,
            "actors",
            _RESERVOIR_ORDER,
        )

        self._validate_keys(
            action_mappers,
            "action_mappers",
            _RESERVOIR_ORDER,
        )

        self.actors = tuple(
            actors[
                reservoir_id
            ]
            for reservoir_id
            in _RESERVOIR_ORDER
        )

        self.action_mappers = tuple(
            action_mappers[
                reservoir_id
            ]
            for reservoir_id
            in _RESERVOIR_ORDER
        )

        # S501 uses independent reservoir-specific Actors.
        if len(
            {
                id(
                    actor
                )
                for actor
                in self.actors
            }
        ) != _NUM_RESERVOIRS:
            raise ValueError(
                "WDD, BHT, XLD, XJB and THR "
                "must use independent Actor objects"
            )

        if len(
            {
                id(
                    mapper
                )
                for mapper
                in self.action_mappers
            }
        ) != _NUM_RESERVOIRS:
            raise ValueError(
                "each reservoir must use its own "
                "ReleaseActionMapper object"
            )

        release_grids = []

        for reservoir_id, actor, mapper in zip(
            _RESERVOIR_ORDER,
            self.actors,
            self.action_mappers,
        ):
            if not isinstance(
                mapper,
                ReleaseActionMapper,
            ):
                raise TypeError(
                    f"{reservoir_id}: "
                    "action mapper must be "
                    "ReleaseActionMapper"
                )

            if (
                str(
                    mapper.reservoir_id
                )
                != reservoir_id
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action mapper "
                    "reservoir ID mismatch"
                )

            releases = np.asarray(
                mapper.candidate_releases_m3s,
                dtype=np.float64,
            )

            expected_shape = (
                int(
                    mapper.num_actions
                ),
            )

            if (
                releases.shape
                != expected_shape
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "candidate release grid "
                    "has invalid shape"
                )

            if not np.all(
                np.isfinite(
                    releases
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "candidate release grid "
                    "contains non-finite values"
                )

            if np.any(
                releases < 0.0
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "candidate release grid "
                    "contains negative values"
                )

            if not np.all(
                np.diff(
                    releases
                )
                > 0.0
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "candidate release grid "
                    "must be strictly increasing"
                )

            releases = releases.copy()

            releases.setflags(
                write=False
            )

            release_grids.append(
                releases
            )

            get_actions = getattr(
                actor,
                "get_actions",
                None,
            )

            if not callable(
                get_actions
            ):
                raise TypeError(
                    f"{reservoir_id}: "
                    "Actor must implement "
                    "get_actions()"
                )

            act_space = getattr(
                actor,
                "act_space",
                None,
            )

            if (
                act_space is None
                or not hasattr(
                    act_space,
                    "n",
                )
            ):
                raise TypeError(
                    f"{reservoir_id}: "
                    "Actor must use a "
                    "Discrete action space"
                )

            try:
                actor_num_actions = int(
                    act_space.n
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise TypeError(
                    f"{reservoir_id}: "
                    "invalid Actor "
                    "Discrete action space"
                ) from exc

            if (
                actor_num_actions
                != mapper.num_actions
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "Actor action space does not "
                    "match action mapper"
                )

        self._release_grids = tuple(
            release_grids
        )

    @torch.no_grad()
    def sample(
        self,
        base_observations: Mapping[
            str,
            np.ndarray,
        ],
        external_inflow_m3s,
        interval_inflows_m3s: Mapping[
            str,
            np.ndarray,
        ],
        safety_results: Sequence[
            SafetyRecoveryResult
        ],
        rnn_states: Mapping[
            str,
            np.ndarray,
        ],
        rnn_masks: Mapping[
            str,
            np.ndarray,
        ],
        deterministic: bool = False,
    ) -> SequentialSamplingResult:
        """Sample one complete cascade action in hydraulic order."""

        if not isinstance(
            deterministic,
            (bool, np.bool_),
        ):
            raise TypeError(
                "deterministic must be boolean"
            )

        deterministic = bool(
            deterministic
        )

        # Conveniently allow the one-environment case to pass
        # one SafetyRecoveryResult directly, while preserving the
        # existing batched Sequence interface.
        if isinstance(
            safety_results,
            SafetyRecoveryResult,
        ):
            safety_results = (
                safety_results,
            )

        else:
            try:
                safety_results = tuple(
                    safety_results
                )

            except TypeError as exc:
                raise TypeError(
                    "safety_results must be a "
                    "SafetyRecoveryResult or "
                    "a sequence of them"
                ) from exc

        batch_size = len(
            safety_results
        )

        if batch_size <= 0:
            raise ValueError(
                "safety_results cannot be empty"
            )

        self._validate_keys(
            base_observations,
            "base_observations",
            _RESERVOIR_ORDER,
        )

        self._validate_keys(
            interval_inflows_m3s,
            "interval_inflows_m3s",
            _DOWNSTREAM_RESERVOIRS,
        )

        self._validate_keys(
            rnn_states,
            "rnn_states",
            _RESERVOIR_ORDER,
        )

        self._validate_keys(
            rnn_masks,
            "rnn_masks",
            _RESERVOIR_ORDER,
        )

        self._validate_safety_results(
            safety_results
        )

        prepared_observations = {
            reservoir_id:
            self._prepare_observation_batch(
                base_observations[
                    reservoir_id
                ],
                batch_size,
                (
                    f"{reservoir_id} "
                    "base_observations"
                ),
            )
            for reservoir_id
            in _RESERVOIR_ORDER
        }

        prepared_rnn_states = {
            reservoir_id:
            self._prepare_rnn_state_batch(
                rnn_states[
                    reservoir_id
                ],
                batch_size,
                (
                    f"{reservoir_id} "
                    "rnn_states"
                ),
            )
            for reservoir_id
            in _RESERVOIR_ORDER
        }

        prepared_rnn_masks = {
            reservoir_id:
            self._prepare_rnn_mask_batch(
                rnn_masks[
                    reservoir_id
                ],
                batch_size,
                (
                    f"{reservoir_id} "
                    "rnn_masks"
                ),
            )
            for reservoir_id
            in _RESERVOIR_ORDER
        }

        external_inflow = (
            self._prepare_flow_batch(
                external_inflow_m3s,
                batch_size,
                "external_inflow_m3s",
            )
        )

        interval_inflows = {
            reservoir_id:
            self._prepare_flow_batch(
                interval_inflows_m3s[
                    reservoir_id
                ],
                batch_size,
                (
                    f"{reservoir_id} "
                    "interval_inflow_m3s"
                ),
            )
            for reservoir_id
            in _DOWNSTREAM_RESERVOIRS
        }

        actions = np.empty(
            (
                batch_size,
                _NUM_RESERVOIRS,
                1,
            ),
            dtype=np.int64,
        )

        action_log_probs = np.empty(
            (
                batch_size,
                _NUM_RESERVOIRS,
                1,
            ),
            dtype=np.float32,
        )

        inflows_m3s = np.empty(
            (
                batch_size,
                _NUM_RESERVOIRS,
            ),
            dtype=np.float64,
        )

        target_releases_m3s = (
            np.empty(
                (
                    batch_size,
                    _NUM_RESERVOIRS,
                ),
                dtype=np.float64,
            )
        )

        executed_releases_m3s = (
            np.empty(
                (
                    batch_size,
                    _NUM_RESERVOIRS,
                ),
                dtype=np.float64,
            )
        )

        decision_observations = []

        action_masks = []

        next_rnn_states = []

        for reservoir_index, reservoir_id in enumerate(
            _RESERVOIR_ORDER
        ):
            actor = self.actors[
                reservoir_index
            ]

            mapper = self.action_mappers[
                reservoir_index
            ]

            if reservoir_index == 0:
                current_inflow = (
                    external_inflow.copy()
                )

            else:
                # The directly upstream reservoir has already
                # been sampled in this same S4 pass.
                #
                # Under the current implementation:
                #
                #     Q_exec = Q_target
                #
                # so this value is exactly the release that S202
                # used when defining candidate compatibility.
                upstream_release = (
                    executed_releases_m3s[
                        :,
                        reservoir_index - 1,
                    ]
                )

                current_inflow = (
                    self._compute_downstream_inflow_batch(
                        upstream_release_m3s=(
                            upstream_release
                        ),
                        interval_inflow_m3s=(
                            interval_inflows[
                                reservoir_id
                            ]
                        ),
                        reservoir_id=(
                            reservoir_id
                        ),
                    )
                )

            inflows_m3s[
                :,
                reservoir_index,
            ] = current_inflow

            decision_observation = (
                self._build_decision_observation(
                    reservoir_id=(
                        reservoir_id
                    ),
                    base_observation=(
                        prepared_observations[
                            reservoir_id
                        ]
                    ),
                    current_inflow_m3s=(
                        current_inflow
                    ),
                    inflow_scale_m3s=mapper.map_max_release_m3s,
                )
            )

            self._validate_actor_observation(
                actor=actor,
                reservoir_id=(
                    reservoir_id
                ),
                observation=(
                    decision_observation
                ),
            )

            if reservoir_index == 0:
                upstream_actions = None

            else:
                upstream_actions = actions[
                    :,
                    reservoir_index - 1,
                    0,
                ]

            action_mask = (
                self._build_action_mask(
                    reservoir_id=(
                        reservoir_id
                    ),
                    reservoir_index=(
                        reservoir_index
                    ),
                    safety_results=(
                        safety_results
                    ),
                    upstream_actions=(
                        upstream_actions
                    ),
                    num_actions=(
                        mapper.num_actions
                    ),
                )
            )

            # Keep exact copies of the behavior conditions that
            # produced this action.  These are the o_dec and M
            # that must later be written into ActorBuffer.
            behavior_observation = (
                decision_observation.copy()
            )

            behavior_mask = (
                action_mask.copy()
            )

            actor_output = (
                actor.get_actions(
                    decision_observation,
                    prepared_rnn_states[
                        reservoir_id
                    ],
                    prepared_rnn_masks[
                        reservoir_id
                    ],
                    action_mask,
                    deterministic=(
                        deterministic
                    ),
                )
            )

            if (
                not isinstance(
                    actor_output,
                    (tuple, list),
                )
                or len(
                    actor_output
                )
                != 3
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "Actor.get_actions() must "
                    "return "
                    "(action, log_prob, rnn_state)"
                )

            (
                sampled_action,
                sampled_log_prob,
                next_rnn_state,
            ) = actor_output

            sampled_action = (
                self._to_numpy(
                    sampled_action
                )
            )

            sampled_log_prob = (
                self._to_numpy(
                    sampled_log_prob
                )
            )

            next_rnn_state = (
                self._to_numpy(
                    next_rnn_state
                )
            )

            action_indices = (
                self._validate_sampled_actions(
                    reservoir_id=(
                        reservoir_id
                    ),
                    sampled_action=(
                        sampled_action
                    ),
                    action_mask=(
                        action_mask
                    ),
                    batch_size=(
                        batch_size
                    ),
                )
            )

            sampled_log_prob = (
                self._prepare_log_prob_batch(
                    sampled_log_prob,
                    batch_size,
                    reservoir_id,
                )
            )

            next_rnn_state = (
                self._prepare_returned_rnn_state(
                    next_rnn_state,
                    expected_shape=(
                        prepared_rnn_states[
                            reservoir_id
                        ].shape
                    ),
                    reservoir_id=(
                        reservoir_id
                    ),
                )
            )

            candidate_releases = (
                self._release_grids[
                    reservoir_index
                ]
            )

            target_release = (
                candidate_releases[
                    action_indices
                ]
            )

            # Current S4 execution convention.
            #
            # No post-sampling clipping, projection or remapping
            # is allowed here.  All physical feasibility has
            # already been handled by S201-S203/S3.
            executed_release = (
                target_release.copy()
            )

            if not np.array_equal(
                target_release,
                executed_release,
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "Q_exec must equal Q_target "
                    "in the current implementation"
                )

            actions[
                :,
                reservoir_index,
                0,
            ] = action_indices

            action_log_probs[
                :,
                reservoir_index,
                :,
            ] = sampled_log_prob

            target_releases_m3s[
                :,
                reservoir_index,
            ] = target_release

            executed_releases_m3s[
                :,
                reservoir_index,
            ] = executed_release

            decision_observations.append(
                behavior_observation
            )

            action_masks.append(
                behavior_mask
            )

            next_rnn_states.append(
                next_rnn_state
            )

        return SequentialSamplingResult(
            reservoir_order=(
                _RESERVOIR_ORDER
            ),
            decision_observations=tuple(
                decision_observations
            ),
            action_masks=tuple(
                action_masks
            ),
            actions=actions,
            action_log_probs=(
                action_log_probs
            ),
            inflows_m3s=(
                inflows_m3s
            ),
            target_releases_m3s=(
                target_releases_m3s
            ),
            executed_releases_m3s=(
                executed_releases_m3s
            ),
            rnn_states=tuple(
                next_rnn_states
            ),
        )

    def _build_action_mask(
        self,
        *,
        reservoir_id: str,
        reservoir_index: int,
        safety_results,
        upstream_actions,
        num_actions: int,
    ) -> np.ndarray:
        """Build the actual HARL-facing mask used by one Actor."""

        batch_size = len(
            safety_results
        )

        masks = np.empty(
            (
                batch_size,
                num_actions,
            ),
            dtype=np.float32,
        )

        if reservoir_index == 0:
            if upstream_actions is not None:
                raise ValueError(
                    "WDD cannot receive "
                    "upstream actions"
                )

        else:
            upstream_actions = np.asarray(
                upstream_actions
            )

            if upstream_actions.shape != (
                batch_size,
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "upstream_actions has "
                    "invalid shape"
                )

        for thread_id, safety_result in enumerate(
            safety_results
        ):
            if reservoir_index == 0:
                mask = (
                    safety_result
                    .get_action_mask(
                        reservoir_id
                    )
                )

            else:
                upstream_action = int(
                    upstream_actions[
                        thread_id
                    ]
                )

                mask = (
                    safety_result
                    .get_action_mask(
                        reservoir_id,
                        upstream_action_index=(
                            upstream_action
                        ),
                    )
                )

            mask = np.asarray(
                mask
            )

            if (
                mask.shape
                != (
                    num_actions,
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "safety mask has "
                    "invalid shape; "
                    f"expected {(num_actions,)}, "
                    f"received {mask.shape}"
                )

            if not np.all(
                (
                    mask == 0
                )
                | (
                    mask == 1
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "safety mask must "
                    "be binary"
                )

            if not np.any(
                mask
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "empty action mask "
                    "reached S4"
                )

            masks[
                thread_id,
                :,
            ] = mask.astype(
                np.float32,
                copy=False,
            )

        return masks

    @staticmethod
    def _build_decision_observation(
        *,
        reservoir_id: str,
        base_observation: np.ndarray,
        current_inflow_m3s: np.ndarray,
        inflow_scale_m3s: float,
    ) -> np.ndarray:
        """Construct the actual observation used at behavior time."""

        base_observation = np.asarray(
            base_observation,
            dtype=np.float32,
        )

        current_inflow = np.asarray(
            current_inflow_m3s,
            dtype=np.float64,
        )

        if base_observation.ndim != 2:
            raise ValueError(
                f"{reservoir_id}: "
                "base observation must "
                "be two-dimensional"
            )

        if (
            current_inflow.ndim != 1
            or current_inflow.shape[0]
            != base_observation.shape[0]
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "current inflow batch "
                "does not match observation"
            )

        if not np.all(
            np.isfinite(
                current_inflow
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "current inflow contains "
                "non-finite values"
            )

        if np.any(
            current_inflow < 0.0
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "current inflow cannot "
                "be negative"
            )

        if reservoir_id == "WDD":
            return (
                base_observation.copy()
            )

        # Match the existing base-observation flow scale. Appending raw
        # m3/s beside normalized features makes input LayerNorm suppress
        # water-level, storage and seasonal information. This changes only
        # the feature encoding; inflows_m3s retains the physical flow.
        inflow_scale = float(inflow_scale_m3s)
        if not np.isfinite(inflow_scale) or inflow_scale <= 0.0:
            raise ValueError("inflow_scale_m3s must be finite and positive")

        inflow_feature = (
            (current_inflow / inflow_scale)[
                :,
                np.newaxis,
            ].astype(
                np.float32
            )
        )

        observation = np.concatenate(
            (
                base_observation,
                inflow_feature,
            ),
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        if not np.all(
            np.isfinite(
                observation
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "decision observation "
                "contains non-finite values"
            )

        return observation

    @staticmethod
    def _validate_sampled_actions(
        *,
        reservoir_id: str,
        sampled_action: np.ndarray,
        action_mask: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        sampled_action = np.asarray(
            sampled_action
        )

        if (
            sampled_action.dtype
            == np.bool_
        ):
            raise TypeError(
                f"{reservoir_id}: "
                "Actor returned "
                "boolean actions"
            )

        if (
            not np.issubdtype(
                sampled_action.dtype,
                np.number,
            )
            or np.issubdtype(
                sampled_action.dtype,
                np.complexfloating,
            )
        ):
            raise TypeError(
                f"{reservoir_id}: "
                "Actor actions must "
                "be real numeric values"
            )

        if sampled_action.ndim == 1:
            sampled_action = (
                sampled_action[
                    :,
                    np.newaxis,
                ]
            )

        if (
            sampled_action.shape
            != (
                batch_size,
                1,
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "Actor must return "
                "one discrete action "
                "per environment"
            )

        action_values = (
            sampled_action[
                :,
                0
            ]
        )

        if not np.all(
            np.isfinite(
                action_values
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "Actor returned "
                "non-finite actions"
            )

        action_indices = (
            action_values.astype(
                np.int64
            )
        )

        if not np.all(
            action_values
            == action_indices
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "Actor returned "
                "non-integer actions"
            )

        if np.any(
            action_indices < 0
        ) or np.any(
            action_indices
            >= action_mask.shape[
                1
            ]
        ):
            raise IndexError(
                f"{reservoir_id}: "
                "sampled action is "
                "outside action space"
            )

        allowed = action_mask[
            np.arange(
                batch_size
            ),
            action_indices,
        ]

        if np.any(
            allowed <= 0.0
        ):
            raise RuntimeError(
                f"{reservoir_id}: "
                "Actor sampled a "
                "masked action"
            )

        return action_indices

    @staticmethod
    def _prepare_observation_batch(
        values,
        batch_size: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim == 1:
            if batch_size != 1:
                raise ValueError(
                    f"{name} must include "
                    "a batch dimension"
                )

            array = array[
                np.newaxis,
                :
            ]

        if array.ndim != 2:
            raise ValueError(
                f"{name} must be "
                "two-dimensional"
            )

        if (
            array.shape[0]
            != batch_size
        ):
            raise ValueError(
                f"{name} has invalid "
                "batch dimension"
            )

        if (
            array.shape[1]
            <= 0
        ):
            raise ValueError(
                f"{name} cannot "
                "be empty"
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

        return array.copy()

    @staticmethod
    def _prepare_rnn_state_batch(
        values,
        batch_size: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    f"{name} must include "
                    "a batch dimension"
                )

            array = array[
                np.newaxis,
                ...
            ]

        if array.ndim != 3:
            raise ValueError(
                f"{name} must have shape "
                "(batch, recurrent_n, "
                "hidden_size)"
            )

        if (
            array.shape[0]
            != batch_size
        ):
            raise ValueError(
                f"{name} has invalid "
                "batch dimension"
            )

        if (
            array.shape[1] <= 0
            or array.shape[2] <= 0
        ):
            raise ValueError(
                f"{name} has invalid "
                "recurrent dimensions"
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

        return array.copy()

    @staticmethod
    def _prepare_rnn_mask_batch(
        values,
        batch_size: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    f"{name} must include "
                    "a batch dimension"
                )

            array = array.reshape(
                1,
                1,
            )

        elif array.ndim == 1:
            if (
                array.size
                != batch_size
            ):
                raise ValueError(
                    f"{name} has "
                    "invalid shape"
                )

            array = array[
                :,
                np.newaxis,
            ]

        if (
            array.shape
            != (
                batch_size,
                1,
            )
        ):
            raise ValueError(
                f"{name} must have shape "
                f"({batch_size}, 1)"
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

        if not np.all(
            (
                array == 0.0
            )
            | (
                array == 1.0
            )
        ):
            raise ValueError(
                f"{name} must be binary"
            )

        return array.copy()

    @staticmethod
    def _prepare_flow_batch(
        values,
        batch_size: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    f"{name} must include "
                    "one value per "
                    "environment"
                )

            array = array.reshape(
                1
            )

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be "
                "one-dimensional"
            )

        if (
            array.size
            != batch_size
        ):
            raise ValueError(
                f"{name} has invalid "
                "batch dimension"
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

        if np.any(
            array < 0.0
        ):
            raise ValueError(
                f"{name} cannot contain "
                "negative values"
            )

        return array.copy()

    @staticmethod
    def _prepare_log_prob_batch(
        values,
        batch_size: int,
        reservoir_id: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim == 1:
            array = array[
                :,
                np.newaxis,
            ]

        if (
            array.shape
            != (
                batch_size,
                1,
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "action log probability "
                "has invalid shape"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "action log probability "
                "contains non-finite values"
            )

        return array.copy()

    @staticmethod
    def _prepare_returned_rnn_state(
        values,
        *,
        expected_shape,
        reservoir_id: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if (
            array.shape
            != tuple(
                expected_shape
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "Actor returned RNN state "
                f"shape {array.shape}, "
                "expected "
                f"{tuple(expected_shape)}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "Actor returned "
                "non-finite RNN state"
            )

        return array.copy()

    @staticmethod
    def _compute_downstream_inflow_batch(
        *,
        upstream_release_m3s,
        interval_inflow_m3s,
        reservoir_id: str,
    ) -> np.ndarray:
        upstream_release = np.asarray(
            upstream_release_m3s,
            dtype=np.float64,
        )

        interval_inflow = np.asarray(
            interval_inflow_m3s,
            dtype=np.float64,
        )

        if (
            upstream_release.ndim != 1
            or interval_inflow.ndim != 1
            or upstream_release.shape
            != interval_inflow.shape
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "upstream release and "
                "interval inflow batches "
                "must have identical "
                "one-dimensional shapes"
            )

        if not np.all(
            np.isfinite(
                upstream_release
            )
        ) or not np.all(
            np.isfinite(
                interval_inflow
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "downstream inflow inputs "
                "contain non-finite values"
            )

        if np.any(
            upstream_release < 0.0
        ) or np.any(
            interval_inflow < 0.0
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "downstream inflow inputs "
                "cannot be negative"
            )

        result = np.empty(
            upstream_release.shape,
            dtype=np.float64,
        )

        for thread_id in range(
            upstream_release.size
        ):
            result[
                thread_id
            ] = float(
                compute_downstream_inflow(
                    upstream_flow=float(
                        upstream_release[
                            thread_id
                        ]
                    ),
                    interval_inflow=float(
                        interval_inflow[
                            thread_id
                        ]
                    ),
                )
            )

        if not np.all(
            np.isfinite(
                result
            )
        ):
            raise RuntimeError(
                f"{reservoir_id}: "
                "downstream inflow "
                "calculation produced "
                "non-finite values"
            )

        if np.any(
            result < 0.0
        ):
            raise RuntimeError(
                f"{reservoir_id}: "
                "downstream inflow "
                "calculation produced "
                "negative values"
            )

        return result

    @staticmethod
    def _validate_actor_observation(
        *,
        actor,
        reservoir_id: str,
        observation: np.ndarray,
    ) -> None:
        observation = np.asarray(
            observation
        )

        if observation.ndim != 2:
            raise ValueError(
                f"{reservoir_id}: "
                "decision observation "
                "must be two-dimensional"
            )

        if not np.all(
            np.isfinite(
                observation
            )
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "decision observation "
                "contains non-finite values"
            )

        obs_space = getattr(
            actor,
            "obs_space",
            None,
        )

        if (
            obs_space is None
            or not hasattr(
                obs_space,
                "shape",
            )
        ):
            return

        expected_shape = tuple(
            obs_space.shape
        )

        actual_shape = tuple(
            observation.shape[
                1:
            ]
        )

        if (
            actual_shape
            != expected_shape
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "decision observation "
                f"shape {actual_shape} "
                "does not match Actor "
                "observation space "
                f"{expected_shape}"
            )

    def _validate_safety_results(
        self,
        safety_results,
    ) -> None:
        expected_num_actions = tuple(
            int(
                mapper.num_actions
            )
            for mapper
            in self.action_mappers
        )

        for batch_index, safety_result in enumerate(
            safety_results
        ):
            if not isinstance(
                safety_result,
                SafetyRecoveryResult,
            ):
                raise TypeError(
                    "safety_results must contain "
                    "SafetyRecoveryResult objects"
                )

            if (
                tuple(
                    safety_result
                    .num_actions
                )
                != expected_num_actions
            ):
                raise ValueError(
                    "safety result action "
                    "dimensions do not match "
                    "action mappers"
                )

            extendability = (
                safety_result
                .extendability
            )

            if (
                tuple(
                    extendability
                    .reservoir_order
                )
                != _RESERVOIR_ORDER
            ):
                raise ValueError(
                    "safety result reservoir "
                    "order does not match S4"
                )

            if (
                tuple(
                    extendability
                    .action_counts
                )
                != expected_num_actions
            ):
                raise ValueError(
                    "extendability action "
                    "dimensions do not match "
                    "action mappers"
                )

            if not (
                extendability
                .has_joint_feasible_action
            ):
                raise RuntimeError(
                    "S4 received a safety result "
                    "without a complete "
                    "cascade-feasible action path"
                )

            # In P2 fallback SafetyRecoveryResult.get_action_mask()
            # deliberately exposes only the forced one-hot action.
            #
            # Independently verify that the forced joint action
            # still forms a valid path in the stored P1-only
            # ExtendabilityResult.
            forced = (
                safety_result
                .forced_joint_action
            )

            if forced is not None:
                previous_action = None

                for reservoir_index, reservoir_id in enumerate(
                    _RESERVOIR_ORDER
                ):
                    action = int(
                        forced[
                            reservoir_index
                        ]
                    )

                    if reservoir_index == 0:
                        underlying_mask = (
                            extendability
                            .get_action_mask(
                                reservoir_id
                            )
                        )

                    else:
                        underlying_mask = (
                            extendability
                            .get_action_mask(
                                reservoir_id,
                                upstream_action_index=(
                                    previous_action
                                ),
                            )
                        )

                    underlying_mask = np.asarray(
                        underlying_mask,
                        dtype=bool,
                    )

                    if (
                        underlying_mask.shape
                        != (
                            expected_num_actions[
                                reservoir_index
                            ],
                        )
                    ):
                        raise RuntimeError(
                            "P2 fallback "
                            "extendability mask "
                            "has invalid shape"
                        )

                    if not bool(
                        underlying_mask[
                            action
                        ]
                    ):
                        raise RuntimeError(
                            "P2 fallback forced "
                            "joint action is not "
                            "P1-cascade-feasible"
                        )

                    previous_action = (
                        action
                    )

            # Verify the actual S4-facing WDD mask immediately.
            root_mask = np.asarray(
                safety_result
                .get_action_mask(
                    "WDD"
                )
            )

            if (
                root_mask.shape
                != (
                    expected_num_actions[
                        0
                    ],
                )
            ):
                raise ValueError(
                    f"safety_results[{batch_index}]: "
                    "WDD mask has invalid shape"
                )

            if not np.any(
                root_mask
            ):
                raise RuntimeError(
                    f"safety_results[{batch_index}]: "
                    "empty WDD mask reached S4"
                )

    @staticmethod
    def _validate_keys(
        mapping,
        name: str,
        expected_keys,
    ) -> None:
        if not isinstance(
            mapping,
            Mapping,
        ):
            raise TypeError(
                f"{name} must be a mapping"
            )

        actual = set(
            mapping
        )

        expected = set(
            expected_keys
        )

        if actual != expected:
            raise ValueError(
                f"{name} keys must be "
                f"{expected}; received "
                f"{actual}"
            )

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
