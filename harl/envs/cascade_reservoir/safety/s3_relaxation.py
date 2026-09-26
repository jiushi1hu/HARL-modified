from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from ..action_mapping import ReleaseActionMapper
from ..reservoir_physics import ReservoirPhysics
from .constraint_context import (
    ConstraintContext,
    LevelBounds,
    P1Constraints,
    P2Constraints,
    P3Constraints,
    ReleaseBounds,
)
from .s201_local_feasibility import (
    LocalFeasibilityResult,
    evaluate_local_feasibility,
)
from .s202_compatibility import (
    CompatibilityResult,
    build_compatibility_matrix,
)
from .s203_extendability import (
    ExtendabilityResult,
    compute_extendability,
)


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)


@dataclass(frozen=True)
class ReservoirSafetyState:
    reservoir_id: str
    physics: ReservoirPhysics
    mapper: ReleaseActionMapper
    storage_m3: float
    forcing_inflow_m3s: float
    previous_release_m3s: float
    p1: P1Constraints
    p2: Optional[P2Constraints]


@dataclass(frozen=True)
class SafetyRecoveryResult:
    mode: str
    p4_release_inflow_ratio: float
    p3_relaxation_fraction: float
    extendability: Optional[ExtendabilityResult]
    forced_joint_action: Optional[Tuple[int, ...]]
    p2_total_violation: float
    num_actions: Tuple[int, ...]

    @property
    def uses_forced_joint_action(self) -> bool:
        return self.forced_joint_action is not None

    def get_action_mask(
        self,
        reservoir_id: str,
        upstream_action_index: Optional[int] = None,
    ) -> np.ndarray:
        if reservoir_id not in _RESERVOIR_ORDER:
            raise KeyError(
                f"unknown reservoir_id: {reservoir_id}"
            )

        reservoir_index = _RESERVOIR_ORDER.index(
            reservoir_id
        )

        if self.forced_joint_action is not None:
            mask = np.zeros(
                self.num_actions[reservoir_index],
                dtype=bool,
            )

            mask[
                self.forced_joint_action[
                    reservoir_index
                ]
            ] = True

            return mask

        if self.extendability is None:
            raise RuntimeError(
                "extendability result is unavailable"
            )

        return self.extendability.get_action_mask(
            reservoir_id=reservoir_id,
            upstream_action_index=(
                upstream_action_index
            ),
        )


@dataclass(frozen=True)
class _S2Evaluation:
    wdd_local_result: LocalFeasibilityResult
    compatibility_results: Tuple[
        CompatibilityResult,
        ...
    ]
    extendability: ExtendabilityResult


class S3RelaxationSolver:

    def __init__(
        self,
        safety_config: Mapping,
        timestep_seconds: float,
    ):
        p3 = safety_config["p3"]
        p4 = safety_config["p4"]
        relaxation = safety_config[
            "relaxation"
        ]

        self.timestep_seconds = (
            self._validate_positive(
                timestep_seconds,
                "timestep_seconds",
            )
        )

        self.release_change_ratio = (
            self._validate_nonnegative(
                p3["release_change_ratio"],
                "release_change_ratio",
            )
        )

        self.release_change_floor_m3s = (
            self._validate_nonnegative(
                p3[
                    "release_change_floor_m3s"
                ],
                "release_change_floor_m3s",
            )
        )

        self.p3_max_factor = (
            self._validate_positive(
                p3["max_relaxation_factor"],
                "max_relaxation_factor",
            )
        )

        if self.p3_max_factor < 1.0:
            raise ValueError(
                "max_relaxation_factor "
                "must be at least 1"
            )

        level_limits = p3[
            "level_change_limit_m"
        ]

        if set(level_limits) != set(
            _RESERVOIR_ORDER
        ):
            raise ValueError(
                "level_change_limit_m must "
                "contain WDD, BHT, XLD, XJB, THR"
            )

        self.level_change_limits_m = {
            reservoir_id:
            self._validate_positive(
                level_limits[reservoir_id],
                (
                    f"{reservoir_id} "
                    "level_change_limit_m"
                ),
            )
            for reservoir_id
            in _RESERVOIR_ORDER
        }

        self.p4_normal_ratio = (
            self._validate_nonnegative(
                p4[
                    "normal_release_inflow_ratio"
                ],
                "normal_release_inflow_ratio",
            )
        )

        self.p4_min_ratio = (
            self._validate_nonnegative(
                p4[
                    "min_release_inflow_ratio"
                ],
                "min_release_inflow_ratio",
            )
        )

        if (
            self.p4_min_ratio
            > self.p4_normal_ratio
        ):
            raise ValueError(
                "min_release_inflow_ratio "
                "cannot exceed normal ratio"
            )

        self.search_tolerance = (
            self._validate_positive(
                relaxation["tolerance"],
                "relaxation tolerance",
            )
        )

        self.max_iterations = int(
            relaxation["max_iterations"]
        )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive"
            )

    def solve(
        self,
        states: Mapping[
            str,
            ReservoirSafetyState,
        ],
        operation_stage: str,
    ) -> SafetyRecoveryResult:

        states = self._validate_states(
            states
        )

        p4_active = (
            operation_stage == "dry_supply"
        )

        normal = self._evaluate_s2(
            states=states,
            p4_ratio=self.p4_normal_ratio,
            p3_relaxation_fraction=0.0,
            apply_p2=True,
            p4_active=p4_active,
        )

        if (
            normal.extendability
            .has_joint_feasible_action
        ):
            return self._build_result(
                states=states,
                mode="normal",
                p4_ratio=self.p4_normal_ratio,
                p3_fraction=0.0,
                evaluation=normal,
            )

        p4_ratio = self.p4_normal_ratio

        if p4_active:
            p4_max_relaxed = (
                self._evaluate_s2(
                    states=states,
                    p4_ratio=(
                        self.p4_min_ratio
                    ),
                    p3_relaxation_fraction=0.0,
                    apply_p2=True,
                    p4_active=True,
                )
            )

            if (
                p4_max_relaxed
                .extendability
                .has_joint_feasible_action
            ):
                (
                    p4_ratio,
                    evaluation,
                ) = self._search_p4(
                    states=states,
                )

                return self._build_result(
                    states=states,
                    mode="p4_relaxed",
                    p4_ratio=p4_ratio,
                    p3_fraction=0.0,
                    evaluation=evaluation,
                )

            p4_ratio = self.p4_min_ratio

        p3_max_relaxed = self._evaluate_s2(
            states=states,
            p4_ratio=p4_ratio,
            p3_relaxation_fraction=1.0,
            apply_p2=True,
            p4_active=p4_active,
        )

        if (
            p3_max_relaxed
            .extendability
            .has_joint_feasible_action
        ):
            (
                p3_fraction,
                evaluation,
            ) = self._search_p3(
                states=states,
                p4_ratio=p4_ratio,
                p4_active=p4_active,
            )

            return self._build_result(
                states=states,
                mode="p3_relaxed",
                p4_ratio=p4_ratio,
                p3_fraction=p3_fraction,
                evaluation=evaluation,
            )

        return self._solve_p2_min_violation(
            states=states,
            p4_ratio=p4_ratio,
            p4_active=p4_active,
        )

    def _search_p4(
        self,
        states,
    ):
        feasible_ratio = self.p4_min_ratio
        infeasible_ratio = (
            self.p4_normal_ratio
        )

        best_evaluation = self._evaluate_s2(
            states=states,
            p4_ratio=feasible_ratio,
            p3_relaxation_fraction=0.0,
            apply_p2=True,
            p4_active=True,
        )

        for _ in range(
            self.max_iterations
        ):
            if (
                infeasible_ratio
                - feasible_ratio
                <= self.search_tolerance
            ):
                break

            middle = (
                feasible_ratio
                + infeasible_ratio
            ) / 2.0

            evaluation = self._evaluate_s2(
                states=states,
                p4_ratio=middle,
                p3_relaxation_fraction=0.0,
                apply_p2=True,
                p4_active=True,
            )

            if (
                evaluation.extendability
                .has_joint_feasible_action
            ):
                feasible_ratio = middle
                best_evaluation = evaluation
            else:
                infeasible_ratio = middle

        return (
            feasible_ratio,
            best_evaluation,
        )

    def _search_p3(
        self,
        states,
        p4_ratio,
        p4_active,
    ):
        infeasible_fraction = 0.0
        feasible_fraction = 1.0

        best_evaluation = self._evaluate_s2(
            states=states,
            p4_ratio=p4_ratio,
            p3_relaxation_fraction=(
                feasible_fraction
            ),
            apply_p2=True,
            p4_active=p4_active,
        )

        for _ in range(
            self.max_iterations
        ):
            if (
                feasible_fraction
                - infeasible_fraction
                <= self.search_tolerance
            ):
                break

            middle = (
                feasible_fraction
                + infeasible_fraction
            ) / 2.0

            evaluation = self._evaluate_s2(
                states=states,
                p4_ratio=p4_ratio,
                p3_relaxation_fraction=middle,
                apply_p2=True,
                p4_active=p4_active,
            )

            if (
                evaluation.extendability
                .has_joint_feasible_action
            ):
                feasible_fraction = middle
                best_evaluation = evaluation
            else:
                infeasible_fraction = middle

        return (
            feasible_fraction,
            best_evaluation,
        )

    def _evaluate_s2(
        self,
        states,
        p4_ratio,
        p3_relaxation_fraction,
        apply_p2,
        p4_active,
    ):
        wdd_state = states["WDD"]

        wdd_local_result = (
            self._evaluate_local(
                state=wdd_state,
                inflow_m3s=(
                    wdd_state
                    .forcing_inflow_m3s
                ),
                p4_ratio=p4_ratio,
                p3_relaxation_fraction=(
                    p3_relaxation_fraction
                ),
                apply_p2=apply_p2,
                p4_active=p4_active,
            )
        )

        compatibility_results = []

        for index in range(
            1,
            len(_RESERVOIR_ORDER),
        ):
            upstream_id = (
                _RESERVOIR_ORDER[
                    index - 1
                ]
            )
            downstream_id = (
                _RESERVOIR_ORDER[
                    index
                ]
            )

            upstream_state = states[
                upstream_id
            ]
            downstream_state = states[
                downstream_id
            ]

            def evaluate_downstream(
                candidate_inflow_m3s,
                state=downstream_state,
            ):
                return self._evaluate_local(
                    state=state,
                    inflow_m3s=(
                        candidate_inflow_m3s
                    ),
                    p4_ratio=p4_ratio,
                    p3_relaxation_fraction=(
                        p3_relaxation_fraction
                    ),
                    apply_p2=apply_p2,
                    p4_active=p4_active,
                )

            compatibility = (
                build_compatibility_matrix(
                    upstream_mapper=(
                        upstream_state.mapper
                    ),
                    downstream_mapper=(
                        downstream_state.mapper
                    ),
                    interval_inflow_m3s=(
                        downstream_state
                        .forcing_inflow_m3s
                    ),
                    evaluate_downstream=(
                        evaluate_downstream
                    ),
                )
            )

            compatibility_results.append(
                compatibility
            )

        compatibility_results = tuple(
            compatibility_results
        )

        extendability = (
            compute_extendability(
                wdd_local_mask=(
                    wdd_local_result
                    .action_mask
                ),
                compatibility_results=(
                    compatibility_results
                ),
            )
        )

        return _S2Evaluation(
            wdd_local_result=(
                wdd_local_result
            ),
            compatibility_results=(
                compatibility_results
            ),
            extendability=extendability,
        )

    def _evaluate_local(
        self,
        state,
        inflow_m3s,
        p4_ratio,
        p3_relaxation_fraction,
        apply_p2,
        p4_active,
    ):
        context = self._build_context(
            state=state,
            p3_relaxation_fraction=(
                p3_relaxation_fraction
            ),
            apply_p2=apply_p2,
        )

        bounds = context.resolve_s201_bounds(
            apply_p2=apply_p2,
            apply_p3=True,
        )

        result = evaluate_local_feasibility(
            physics=state.physics,
            storage_m3=state.storage_m3,
            inflow_m3s=inflow_m3s,
            bounds=bounds,
            candidate_releases_m3s=(
                state.mapper
                .candidate_releases_m3s
            ),
            timestep_seconds=(
                self.timestep_seconds
            ),
        )

        if not p4_active:
            return result

        candidate_releases = (
            state.mapper
            .candidate_releases_m3s
        )

        p4_mask = (
            candidate_releases
            >= p4_ratio * inflow_m3s
            - 1e-9
        )

        action_mask = (
            result.action_mask
            & p4_mask
        )

        action_mask.setflags(
            write=False
        )

        return LocalFeasibilityResult(
            safe_min_release_m3s=(
                result
                .safe_min_release_m3s
            ),
            safe_max_release_m3s=(
                result
                .safe_max_release_m3s
            ),
            action_mask=action_mask,
        )

    def _build_context(
        self,
        state,
        p3_relaxation_fraction,
        apply_p2,
    ):
        if not (
            0.0
            <= p3_relaxation_fraction
            <= 1.0
        ):
            raise ValueError(
                "p3_relaxation_fraction must "
                "be in [0, 1]"
            )

        multiplier = (
            1.0
            + p3_relaxation_fraction
            * (
                self.p3_max_factor
                - 1.0
            )
        )

        current_level = (
            state.physics.storage_to_level(
                state.storage_m3
            )
        )

        level_change_limit = (
            self.level_change_limits_m[
                state.reservoir_id
            ]
            * multiplier
        )

        release_change_limit = max(
            self.release_change_ratio
            * state.previous_release_m3s,
            self.release_change_floor_m3s,
        ) * multiplier

        p3 = P3Constraints(
            level=LevelBounds(
                min_level_m=(
                    current_level
                    - level_change_limit
                ),
                max_level_m=(
                    current_level
                    + level_change_limit
                ),
            ),
            release=ReleaseBounds(
                min_release_m3s=max(
                    0.0,
                    state.previous_release_m3s
                    - release_change_limit,
                ),
                max_release_m3s=(
                    state.previous_release_m3s
                    + release_change_limit
                ),
            ),
        )

        return ConstraintContext(
            reservoir_id=(
                state.reservoir_id
            ),
            p1=state.p1,
            p2=(
                state.p2
                if apply_p2
                else None
            ),
            p3=p3,
        )

    def _solve_p2_min_violation(
        self,
        states,
        p4_ratio,
        p4_active,
    ):
        evaluation = self._evaluate_s2(
            states=states,
            p4_ratio=p4_ratio,
            p3_relaxation_fraction=1.0,
            apply_p2=False,
            p4_active=p4_active,
        )

        if not (
            evaluation.extendability
            .has_joint_feasible_action
        ):
            raise RuntimeError(
                "No joint action remains after "
                "maximum P3/P4 relaxation "
                "while P1 is enforced"
            )

        wdd_state = states["WDD"]

        previous_cost = np.full(
            wdd_state.mapper.num_actions,
            np.inf,
            dtype=np.float64,
        )

        wdd_inflow = (
            wdd_state
            .forcing_inflow_m3s
        )

        for action in np.flatnonzero(
            evaluation
            .wdd_local_result
            .action_mask
        ):
            previous_cost[action] = (
                self._p2_action_violation(
                    state=wdd_state,
                    inflow_m3s=wdd_inflow,
                    action_index=int(action),
                )
            )

        parent_tables = []

        for link_index, compatibility in enumerate(
            evaluation.compatibility_results
        ):
            downstream_id = (
                _RESERVOIR_ORDER[
                    link_index + 1
                ]
            )

            downstream_state = states[
                downstream_id
            ]

            matrix = (
                compatibility
                .compatibility_matrix
            )

            edge_cost = np.full(
                matrix.shape,
                np.inf,
                dtype=np.float64,
            )

            for upstream_action in range(
                matrix.shape[0]
            ):
                candidate_inflow = float(
                    compatibility
                    .candidate_inflows_m3s[
                        upstream_action
                    ]
                )

                compatible_actions = (
                    np.flatnonzero(
                        matrix[
                            upstream_action,
                            :,
                        ]
                    )
                )

                for downstream_action in (
                    compatible_actions
                ):
                    edge_cost[
                        upstream_action,
                        downstream_action,
                    ] = (
                        self._p2_action_violation(
                            state=(
                                downstream_state
                            ),
                            inflow_m3s=(
                                candidate_inflow
                            ),
                            action_index=int(
                                downstream_action
                            ),
                        )
                    )

            total_cost = (
                previous_cost[:, np.newaxis]
                + edge_cost
            )

            parents = np.argmin(
                total_cost,
                axis=0,
            )

            downstream_cost = total_cost[
                parents,
                np.arange(
                    total_cost.shape[1]
                ),
            ]

            invalid = ~np.isfinite(
                downstream_cost
            )

            parents = parents.astype(
                np.int64
            )

            parents[invalid] = -1

            parent_tables.append(
                parents
            )

            previous_cost = (
                downstream_cost
            )

        if not np.isfinite(
            previous_cost
        ).any():
            raise RuntimeError(
                "Unable to construct a P1-safe "
                "joint action"
            )

        final_action = int(
            np.argmin(
                previous_cost
            )
        )

        total_violation = float(
            previous_cost[
                final_action
            ]
        )

        joint_action = [
            0
        ] * len(
            _RESERVOIR_ORDER
        )

        joint_action[-1] = (
            final_action
        )

        for link_index in range(
            len(parent_tables) - 1,
            -1,
            -1,
        ):
            downstream_action = (
                joint_action[
                    link_index + 1
                ]
            )

            upstream_action = int(
                parent_tables[
                    link_index
                ][
                    downstream_action
                ]
            )

            if upstream_action < 0:
                raise RuntimeError(
                    "Invalid P2 fallback "
                    "parent chain"
                )

            joint_action[
                link_index
            ] = upstream_action

        return SafetyRecoveryResult(
            mode="p2_min_violation",
            p4_release_inflow_ratio=(
                p4_ratio
            ),
            p3_relaxation_fraction=1.0,
            extendability=None,
            forced_joint_action=tuple(
                joint_action
            ),
            p2_total_violation=(
                total_violation
            ),
            num_actions=self._num_actions(
                states
            ),
        )

    def _p2_action_violation(
        self,
        state,
        inflow_m3s,
        action_index,
    ):
        if state.p2 is None:
            return 0.0

        release = (
            state.mapper.action_to_release(
                action_index
            )
        )

        next_storage = (
            state.physics.compute_next_storage(
                storage_m3=(
                    state.storage_m3
                ),
                inflow_m3s=inflow_m3s,
                release_m3s=release,
                timestep_seconds=(
                    self.timestep_seconds
                ),
            )
        )

        next_level = (
            state.physics.storage_to_level(
                next_storage
            )
        )

        return self._normalized_p2_violation(
            next_level_m=next_level,
            p1=state.p1,
            p2=state.p2,
        )

    @staticmethod
    def _normalized_p2_violation(
        next_level_m,
        p1,
        p2,
    ):
        violation = 0.0

        p1_min = (
            p1.level.min_level_m
        )
        p1_max = (
            p1.level.max_level_m
        )

        p2_min = (
            p2.level.min_level_m
        )
        p2_max = (
            p2.level.max_level_m
        )

        if p2_max is not None:
            exceedance = max(
                0.0,
                next_level_m
                - p2_max,
            )

            if exceedance > 0.0:
                margin = (
                    p1_max
                    - p2_max
                )

                if margin <= 0.0:
                    return np.inf

                violation += (
                    exceedance
                    / margin
                )

        if p2_min is not None:
            shortfall = max(
                0.0,
                p2_min
                - next_level_m,
            )

            if shortfall > 0.0:
                margin = (
                    p2_min
                    - p1_min
                )

                if margin <= 0.0:
                    return np.inf

                violation += (
                    shortfall
                    / margin
                )

        return float(
            violation
        )

    def _build_result(
        self,
        states,
        mode,
        p4_ratio,
        p3_fraction,
        evaluation,
    ):
        return SafetyRecoveryResult(
            mode=mode,
            p4_release_inflow_ratio=(
                p4_ratio
            ),
            p3_relaxation_fraction=(
                p3_fraction
            ),
            extendability=(
                evaluation.extendability
            ),
            forced_joint_action=None,
            p2_total_violation=0.0,
            num_actions=self._num_actions(
                states
            ),
        )

    @staticmethod
    def _num_actions(
        states,
    ):
        return tuple(
            states[
                reservoir_id
            ].mapper.num_actions
            for reservoir_id
            in _RESERVOIR_ORDER
        )

    @staticmethod
    def _validate_states(
        states,
    ):
        expected = set(
            _RESERVOIR_ORDER
        )

        received = set(
            states
        )

        if received != expected:
            raise ValueError(
                f"states must contain "
                f"{expected}"
            )

        validated = {}

        for reservoir_id in (
            _RESERVOIR_ORDER
        ):
            state = states[
                reservoir_id
            ]

            if not isinstance(
                state,
                ReservoirSafetyState,
            ):
                raise TypeError(
                    "states must contain "
                    "ReservoirSafetyState"
                )

            if (
                state.reservoir_id
                != reservoir_id
            ):
                raise ValueError(
                    f"state key {reservoir_id} "
                    "does not match "
                    f"{state.reservoir_id}"
                )

            if (
                state.physics.reservoir_id
                != reservoir_id
            ):
                raise ValueError(
                    f"{reservoir_id}: physics "
                    "reservoir ID mismatch"
                )

            if (
                state.mapper.reservoir_id
                != reservoir_id
            ):
                raise ValueError(
                    f"{reservoir_id}: mapper "
                    "reservoir ID mismatch"
                )

            forcing = float(
                state.forcing_inflow_m3s
            )

            previous_release = float(
                state.previous_release_m3s
            )

            if (
                not np.isfinite(forcing)
                or forcing < 0.0
            ):
                raise ValueError(
                    f"{reservoir_id}: invalid "
                    "forcing inflow"
                )

            if (
                not np.isfinite(
                    previous_release
                )
                or previous_release < 0.0
            ):
                raise ValueError(
                    f"{reservoir_id}: invalid "
                    "previous release"
                )

            state.physics.storage_to_level(
                state.storage_m3
            )

            validated[
                reservoir_id
            ] = state

        return validated

    @staticmethod
    def _validate_nonnegative(
        value,
        name,
    ):
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

        return value

    @staticmethod
    def _validate_positive(
        value,
        name,
    ):
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value <= 0.0:
            raise ValueError(
                f"{name} must be positive"
            )

        return value
