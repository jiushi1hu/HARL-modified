from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from harl.envs.cascade_reservoir.action_mapping import (
    ReleaseActionMapper,
)
from harl.envs.cascade_reservoir.safety.constraint_context import (
    ConstraintContext,
    LevelBounds,
    P1Constraints,
    P2Constraints,
    P3Constraints,
    ReleaseBounds,
)
from harl.envs.cascade_reservoir.safety.s201_local_feasibility import (
    LocalFeasibilityResult,
    evaluate_local_feasibility,
)
from harl.envs.cascade_reservoir.safety.s202_compatibility import (
    CompatibilityResult,
    build_compatibility_matrix,
)
from harl.envs.cascade_reservoir.safety.s203_extendability import (
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

_NUM_RESERVOIRS = len(
    _RESERVOIR_ORDER
)

_FLOW_TOLERANCE_M3S = 1e-7
_LEVEL_TOLERANCE_M = 1e-8


@dataclass(frozen=True)
class ReservoirSafetyState:
    """Current S2/S3 state of one reservoir."""

    reservoir_id: str
    physics: object
    mapper: ReleaseActionMapper

    storage_m3: float

    # WDD:
    #     external upstream inflow
    #
    # BHT/XLD/XJB/THR:
    #     interval inflow only
    forcing_inflow_m3s: float

    # None only on the first simulation day.
    previous_release_m3s: Optional[float]

    p1: P1Constraints
    p2: P2Constraints

    def __post_init__(
        self,
    ):
        reservoir_id = str(
            self.reservoir_id
        )

        if reservoir_id not in (
            _RESERVOIR_ORDER
        ):
            raise ValueError(
                "unknown reservoir_id: "
                f"{reservoir_id}"
            )

        if not isinstance(
            self.mapper,
            ReleaseActionMapper,
        ):
            raise TypeError(
                f"{reservoir_id}: mapper must be "
                "ReleaseActionMapper"
            )

        mapper_reservoir_id = getattr(
            self.mapper,
            "reservoir_id",
            None,
        )

        if mapper_reservoir_id is None:
            raise AttributeError(
                f"{reservoir_id}: mapper must expose "
                "reservoir_id"
            )

        if str(
            mapper_reservoir_id
        ) != reservoir_id:
            raise ValueError(
                f"{reservoir_id}: mapper belongs to "
                f"{mapper_reservoir_id}"
            )

        if not isinstance(
            self.p1,
            P1Constraints,
        ):
            raise TypeError(
                f"{reservoir_id}: p1 must be "
                "P1Constraints"
            )

        if not isinstance(
            self.p2,
            P2Constraints,
        ):
            raise TypeError(
                f"{reservoir_id}: p2 must be "
                "P2Constraints"
            )

        storage = _nonnegative_finite(
            self.storage_m3,
            f"{reservoir_id}.storage_m3",
        )

        forcing = _nonnegative_finite(
            self.forcing_inflow_m3s,
            (
                f"{reservoir_id}."
                "forcing_inflow_m3s"
            ),
        )

        previous_release = (
            self.previous_release_m3s
        )

        if previous_release is not None:
            previous_release = (
                _nonnegative_finite(
                    previous_release,
                    (
                        f"{reservoir_id}."
                        "previous_release_m3s"
                    ),
                )
            )

        if not hasattr(
            self.physics,
            "storage_from_level",
        ):
            raise AttributeError(
                f"{reservoir_id}: physics must "
                "implement storage_from_level()"
            )

        if not hasattr(
            self.physics,
            "level_from_storage",
        ):
            raise AttributeError(
                f"{reservoir_id}: physics must "
                "implement level_from_storage()"
            )

        timestep = getattr(
            self.physics,
            "timestep_seconds",
            None,
        )

        if timestep is None:
            raise AttributeError(
                f"{reservoir_id}: physics must expose "
                "timestep_seconds"
            )

        _positive_finite(
            timestep,
            (
                f"{reservoir_id}."
                "physics.timestep_seconds"
            ),
        )

        # Validate that the current storage lies inside the
        # physical level-storage interpolation domain.
        self.physics.level_from_storage(
            storage
        )

        object.__setattr__(
            self,
            "reservoir_id",
            reservoir_id,
        )

        object.__setattr__(
            self,
            "storage_m3",
            storage,
        )

        object.__setattr__(
            self,
            "forcing_inflow_m3s",
            forcing,
        )

        object.__setattr__(
            self,
            "previous_release_m3s",
            previous_release,
        )


@dataclass(frozen=True)
class SafetyRecoveryResult:
    """Final S2/S3 executable-action result for one decision step."""

    mode: str

    # None when P4 is inactive.
    p4_release_inflow_ratio: Optional[
        float
    ]

    # 0 = normal P3
    # 1 = maximum configured P3 relaxation
    p3_relaxation_fraction: float

    # Normal / P4 / P3 modes:
    #     S203 result under the recovered constraints.
    #
    # P2 fallback:
    #     P1-only S203 result.
    extendability: ExtendabilityResult

    # Present only in P2 fallback.
    forced_joint_action: Optional[
        Tuple[int, ...]
    ]

    p2_total_violation: float

    # One action dimension per reservoir.
    num_actions: Tuple[int, ...]

    def __post_init__(
        self,
    ):
        valid_modes = {
            "normal",
            "p4_relaxed",
            "p3_relaxed",
            "p2_fallback",
        }

        mode = str(
            self.mode
        )

        if mode not in valid_modes:
            raise ValueError(
                "unknown safety recovery mode: "
                f"{mode}"
            )

        if not isinstance(
            self.extendability,
            ExtendabilityResult,
        ):
            raise TypeError(
                "extendability must be "
                "ExtendabilityResult"
            )

        p3_fraction = _finite(
            self.p3_relaxation_fraction,
            "p3_relaxation_fraction",
        )

        if not (
            0.0
            <= p3_fraction
            <= 1.0
        ):
            raise ValueError(
                "p3_relaxation_fraction must "
                "be in [0, 1]"
            )

        p2_violation = (
            _nonnegative_finite(
                self.p2_total_violation,
                "p2_total_violation",
            )
        )

        p4_ratio = (
            self.p4_release_inflow_ratio
        )

        if p4_ratio is not None:
            p4_ratio = (
                _nonnegative_finite(
                    p4_ratio,
                    (
                        "p4_release_"
                        "inflow_ratio"
                    ),
                )
            )

        num_actions = tuple(
            _validate_num_actions(
                value,
                (
                    f"num_actions[{index}]"
                ),
            )
            for index, value
            in enumerate(
                self.num_actions
            )
        )

        if len(
            num_actions
        ) != _NUM_RESERVOIRS:
            raise ValueError(
                "num_actions must contain "
                "five action dimensions"
            )

        if (
            num_actions
            != self.extendability.action_counts
        ):
            raise ValueError(
                "num_actions does not match "
                "extendability action dimensions"
            )

        forced = (
            self.forced_joint_action
        )

        if forced is not None:
            forced = tuple(
                _validate_action_index(
                    action_index=value,
                    num_actions=(
                        num_actions[index]
                    ),
                    name=(
                        "forced_joint_action"
                        f"[{index}]"
                    ),
                )
                for index, value
                in enumerate(
                    forced
                )
            )

            if len(
                forced
            ) != _NUM_RESERVOIRS:
                raise ValueError(
                    "forced_joint_action must "
                    "contain five actions"
                )

        if mode == "p2_fallback":
            if forced is None:
                raise ValueError(
                    "p2_fallback requires "
                    "forced_joint_action"
                )
        elif forced is not None:
            raise ValueError(
                "forced_joint_action is only "
                "valid in p2_fallback mode"
            )

        object.__setattr__(
            self,
            "mode",
            mode,
        )

        object.__setattr__(
            self,
            "p3_relaxation_fraction",
            p3_fraction,
        )

        object.__setattr__(
            self,
            "p2_total_violation",
            p2_violation,
        )

        object.__setattr__(
            self,
            "p4_release_inflow_ratio",
            p4_ratio,
        )

        object.__setattr__(
            self,
            "forced_joint_action",
            forced,
        )

        object.__setattr__(
            self,
            "num_actions",
            num_actions,
        )

    @property
    def uses_forced_joint_action(
        self,
    ) -> bool:
        return (
            self.forced_joint_action
            is not None
        )

    def get_action_mask(
        self,
        reservoir_id: str,
        upstream_action_index: Optional[
            int
        ] = None,
    ) -> np.ndarray:
        """Return the final behavior-time action mask."""

        reservoir_id = str(
            reservoir_id
        )

        try:
            reservoir_index = (
                _RESERVOIR_ORDER.index(
                    reservoir_id
                )
            )
        except ValueError as exc:
            raise KeyError(
                "unknown reservoir_id: "
                f"{reservoir_id}"
            ) from exc

        # Emergency P2 fallback already selected one
        # complete minimum-violation joint action.
        if (
            self.forced_joint_action
            is not None
        ):
            mask = np.zeros(
                self.num_actions[
                    reservoir_index
                ],
                dtype=bool,
            )

            mask[
                self.forced_joint_action[
                    reservoir_index
                ]
            ] = True

            return mask

        return (
            self.extendability.get_action_mask(
                reservoir_id=reservoir_id,
                upstream_action_index=(
                    upstream_action_index
                ),
            )
        )


@dataclass(frozen=True)
class _S2Evaluation:
    """Internal S2 result under one particular relaxation state."""

    wdd_local_result: LocalFeasibilityResult

    compatibility_results: Tuple[
        CompatibilityResult,
        ...
    ]

    extendability: ExtendabilityResult


class S3RelaxationSolver:
    """S3 hierarchical feasibility-recovery solver.

    Priority:

        P1 > P2 > P3 > P4

    Recovery order when the normal S2 joint feasible set is empty:

        1. minimally relax P4;
        2. if necessary, minimally relax P3;
        3. if still empty, enforce P1 only and choose the
           P2-minimum-violation joint action.

    P1 is never relaxed.
    """

    def __init__(
        self,
        reservoir_order: Sequence[str],
        p3_config: Mapping,
        p4_config: Mapping,
        relaxation_config: Mapping,
    ):
        self.reservoir_order = tuple(
            str(
                reservoir_id
            )
            for reservoir_id
            in reservoir_order
        )

        if (
            self.reservoir_order
            != _RESERVOIR_ORDER
        ):
            raise ValueError(
                "reservoir_order must be "
                "WDD, BHT, XLD, XJB, THR"
            )

        self.release_change_ratio = (
            _nonnegative_finite(
                p3_config[
                    "release_change_ratio"
                ],
                "release_change_ratio",
            )
        )

        self.release_change_floor_m3s = (
            _nonnegative_finite(
                p3_config[
                    "release_change_floor_m3s"
                ],
                (
                    "release_change_"
                    "floor_m3s"
                ),
            )
        )

        self.max_p3_relaxation_factor = (
            _positive_finite(
                p3_config[
                    "max_relaxation_factor"
                ],
                "max_relaxation_factor",
            )
        )

        if (
            self.max_p3_relaxation_factor
            < 1.0
        ):
            raise ValueError(
                "P3 max_relaxation_factor "
                "must be at least 1"
            )

        level_change_config = (
            p3_config[
                "level_change_limit_m"
            ]
        )

        if set(
            level_change_config
        ) != set(
            self.reservoir_order
        ):
            raise ValueError(
                "P3 level_change_limit_m "
                "must define exactly "
                "WDD, BHT, XLD, XJB, THR"
            )

        self.level_change_limit_m = {
            reservoir_id:
            _positive_finite(
                level_change_config[
                    reservoir_id
                ],
                (
                    "level_change_limit_m."
                    f"{reservoir_id}"
                ),
            )
            for reservoir_id
            in self.reservoir_order
        }

        self.normal_p4_ratio = (
            _nonnegative_finite(
                p4_config[
                    "normal_release_inflow_ratio"
                ],
                (
                    "normal_release_"
                    "inflow_ratio"
                ),
            )
        )

        self.min_p4_ratio = (
            _nonnegative_finite(
                p4_config[
                    "min_release_inflow_ratio"
                ],
                (
                    "min_release_"
                    "inflow_ratio"
                ),
            )
        )

        if (
            self.min_p4_ratio
            > self.normal_p4_ratio
        ):
            raise ValueError(
                "P4 minimum ratio cannot "
                "exceed normal ratio"
            )

        self.relaxation_tolerance = (
            _positive_finite(
                relaxation_config[
                    "tolerance"
                ],
                "relaxation.tolerance",
            )
        )

        max_iterations = (
            relaxation_config[
                "max_iterations"
            ]
        )

        if (
            not isinstance(
                max_iterations,
                Integral,
            )
            or isinstance(
                max_iterations,
                (bool, np.bool_),
            )
        ):
            raise TypeError(
                "relaxation.max_iterations "
                "must be an integer"
            )

        self.max_iterations = int(
            max_iterations
        )

        if self.max_iterations <= 0:
            raise ValueError(
                "relaxation.max_iterations "
                "must be positive"
            )

    def solve(
        self,
        states: Sequence[
            ReservoirSafetyState
        ],
        operation_stage: str,
    ) -> SafetyRecoveryResult:
        """Run normal S2 and, if needed, S3 recovery."""

        states = tuple(
            states
        )

        self._validate_states(
            states
        )

        operation_stage = str(
            operation_stage
        )

        if not operation_stage:
            raise ValueError(
                "operation_stage cannot be empty"
            )

        num_actions = tuple(
            int(
                state.mapper.num_actions
            )
            for state in states
        )

        # P4 is active only during the dry-supply stage.
        p4_ratio = (
            self.normal_p4_ratio
            if operation_stage
            == "dry_supply"
            else None
        )

        # ----------------------------------------------------------
        # Normal S2:
        # P1 + P2 + P3 + P4
        # ----------------------------------------------------------
        normal = self._evaluate_s2(
            states=states,
            use_p2=True,
            p3_factor=1.0,
            p4_ratio=p4_ratio,
        )

        if (
            normal.extendability
            .has_joint_feasible_action
        ):
            return SafetyRecoveryResult(
                mode="normal",
                p4_release_inflow_ratio=(
                    p4_ratio
                ),
                p3_relaxation_fraction=0.0,
                extendability=(
                    normal.extendability
                ),
                forced_joint_action=None,
                p2_total_violation=0.0,
                num_actions=num_actions,
            )

        # ----------------------------------------------------------
        # P4 relaxation:
        #
        # beta_4:
        #     normal -> minimum configured value
        #
        # A smaller beta_4 means a weaker P4 constraint.
        # ----------------------------------------------------------
        p4_for_p3 = p4_ratio

        if (
            p4_ratio is not None
            and self.min_p4_ratio
            < self.normal_p4_ratio
        ):
            maximum_p4_relaxation = (
                self._evaluate_s2(
                    states=states,
                    use_p2=True,
                    p3_factor=1.0,
                    p4_ratio=(
                        self.min_p4_ratio
                    ),
                )
            )

            if (
                maximum_p4_relaxation
                .extendability
                .has_joint_feasible_action
            ):
                (
                    recovered_ratio,
                    recovered,
                ) = (
                    self
                    ._search_minimum_p4_relaxation(
                        states=states,
                        feasible_minimum=(
                            maximum_p4_relaxation
                        ),
                    )
                )

                return SafetyRecoveryResult(
                    mode="p4_relaxed",
                    p4_release_inflow_ratio=(
                        recovered_ratio
                    ),
                    p3_relaxation_fraction=0.0,
                    extendability=(
                        recovered.extendability
                    ),
                    forced_joint_action=None,
                    p2_total_violation=0.0,
                    num_actions=num_actions,
                )

            # P4 has reached its maximum allowed relaxation.
            p4_for_p3 = (
                self.min_p4_ratio
            )

        # ----------------------------------------------------------
        # P3 relaxation:
        #
        # factor:
        #     1.0 -> max_p3_relaxation_factor
        #
        # P4 remains at its maximally relaxed value if it was active.
        # ----------------------------------------------------------
        if (
            self.max_p3_relaxation_factor
            > 1.0
        ):
            maximum_p3_relaxation = (
                self._evaluate_s2(
                    states=states,
                    use_p2=True,
                    p3_factor=(
                        self
                        .max_p3_relaxation_factor
                    ),
                    p4_ratio=(
                        p4_for_p3
                    ),
                )
            )

            if (
                maximum_p3_relaxation
                .extendability
                .has_joint_feasible_action
            ):
                (
                    relaxation_fraction,
                    recovered,
                ) = (
                    self
                    ._search_minimum_p3_relaxation(
                        states=states,
                        p4_ratio=(
                            p4_for_p3
                        ),
                        feasible_maximum=(
                            maximum_p3_relaxation
                        ),
                    )
                )

                return SafetyRecoveryResult(
                    mode="p3_relaxed",
                    p4_release_inflow_ratio=(
                        p4_for_p3
                    ),
                    p3_relaxation_fraction=(
                        relaxation_fraction
                    ),
                    extendability=(
                        recovered.extendability
                    ),
                    forced_joint_action=None,
                    p2_total_violation=0.0,
                    num_actions=num_actions,
                )

        # ----------------------------------------------------------
        # P2 minimum-violation fallback.
        #
        # Patent semantics:
        #     P1 remains absolute.
        #
        # P4 and P3 have already exhausted their allowed relaxation
        # ranges.  We now search the P1-safe joint-action graph and
        # select the complete path having minimum total P2 violation.
        # ----------------------------------------------------------
        (
            forced_joint_action,
            total_p2_violation,
            p1_evaluation,
        ) = self._solve_p2_fallback(
            states
        )

        return SafetyRecoveryResult(
            mode="p2_fallback",
            p4_release_inflow_ratio=(
                p4_for_p3
            ),
            p3_relaxation_fraction=1.0,
            extendability=(
                p1_evaluation
                .extendability
            ),
            forced_joint_action=(
                forced_joint_action
            ),
            p2_total_violation=(
                total_p2_violation
            ),
            num_actions=num_actions,
        )

    def _search_minimum_p4_relaxation(
        self,
        states,
        feasible_minimum,
    ):
        """Find the largest feasible beta_4 below the normal value.

        Because a larger beta_4 is stricter, this corresponds to
        the minimum necessary P4 relaxation.
        """

        feasible_ratio = (
            self.min_p4_ratio
        )

        infeasible_ratio = (
            self.normal_p4_ratio
        )

        best_evaluation = (
            feasible_minimum
        )

        for _ in range(
            self.max_iterations
        ):
            if (
                infeasible_ratio
                - feasible_ratio
                <= self.relaxation_tolerance
            ):
                break

            middle = (
                feasible_ratio
                + infeasible_ratio
            ) / 2.0

            evaluation = (
                self._evaluate_s2(
                    states=states,
                    use_p2=True,
                    p3_factor=1.0,
                    p4_ratio=middle,
                )
            )

            if (
                evaluation
                .extendability
                .has_joint_feasible_action
            ):
                feasible_ratio = middle
                best_evaluation = (
                    evaluation
                )

            else:
                infeasible_ratio = middle

        return (
            float(
                feasible_ratio
            ),
            best_evaluation,
        )

    def _search_minimum_p3_relaxation(
        self,
        states,
        p4_ratio,
        feasible_maximum,
    ):
        """Find the minimum P3 relaxation fraction restoring S2."""

        infeasible_fraction = 0.0
        feasible_fraction = 1.0

        best_evaluation = (
            feasible_maximum
        )

        for _ in range(
            self.max_iterations
        ):
            if (
                feasible_fraction
                - infeasible_fraction
                <= self.relaxation_tolerance
            ):
                break

            middle = (
                infeasible_fraction
                + feasible_fraction
            ) / 2.0

            factor = (
                self._p3_factor_from_fraction(
                    middle
                )
            )

            evaluation = (
                self._evaluate_s2(
                    states=states,
                    use_p2=True,
                    p3_factor=factor,
                    p4_ratio=p4_ratio,
                )
            )

            if (
                evaluation
                .extendability
                .has_joint_feasible_action
            ):
                feasible_fraction = middle
                best_evaluation = (
                    evaluation
                )

            else:
                infeasible_fraction = middle

        return (
            float(
                feasible_fraction
            ),
            best_evaluation,
        )

    def _evaluate_s2(
        self,
        *,
        states,
        use_p2: bool,
        p3_factor: Optional[float],
        p4_ratio: Optional[float],
    ) -> _S2Evaluation:
        """Evaluate one complete S201 -> S202 -> S203 chain."""

        wdd_state = states[
            0
        ]

        wdd_local_result = (
            self._evaluate_local(
                state=wdd_state,
                inflow_m3s=(
                    wdd_state
                    .forcing_inflow_m3s
                ),
                use_p2=use_p2,
                p3_factor=p3_factor,
                p4_ratio=p4_ratio,
            )
        )

        compatibility_results = []

        for downstream_index in range(
            1,
            _NUM_RESERVOIRS,
        ):
            upstream_state = states[
                downstream_index - 1
            ]

            downstream_state = states[
                downstream_index
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
                    use_p2=use_p2,
                    p3_factor=p3_factor,
                    p4_ratio=p4_ratio,
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
                    .feasible_mask
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
            extendability=(
                extendability
            ),
        )

    def _evaluate_local(
        self,
        *,
        state: ReservoirSafetyState,
        inflow_m3s: float,
        use_p2: bool,
        p3_factor: Optional[float],
        p4_ratio: Optional[float],
    ) -> LocalFeasibilityResult:
        """Run the canonical S201 implementation plus dynamic P4."""

        inflow = _nonnegative_finite(
            inflow_m3s,
            (
                f"{state.reservoir_id}."
                "candidate_inflow_m3s"
            ),
        )

        context = (
            self._build_context(
                state=state,
                use_p2=use_p2,
                p3_factor=p3_factor,
            )
        )

        result = (
            evaluate_local_feasibility(
                physics=state.physics,
                mapper=state.mapper,
                storage_m3=(
                    state.storage_m3
                ),
                inflow_m3s=inflow,
                constraints=context,
            )
        )

        if p4_ratio is None:
            return result

        p4_ratio = (
            _nonnegative_finite(
                p4_ratio,
                "p4_ratio",
            )
        )

        # If P1/P2/P3 itself is empty there is no need to
        # construct another result merely to apply P4.
        if result.bounds.is_empty:
            return result

        p4_min_release = (
            p4_ratio
            * inflow
        )

        candidate_releases = (
            result
            .candidate_releases_m3s
        )

        p4_mask = (
            candidate_releases
            >= (
                p4_min_release
                - _FLOW_TOLERANCE_M3S
            )
        )

        feasible_mask = (
            result.feasible_mask
            & p4_mask
        )

        safe_min_release = max(
            float(
                result
                .safe_min_release_m3s
            ),
            p4_min_release,
        )

        # Rebuild the canonical result object so S202 continues
        # to receive exactly the same S201 result type.
        return LocalFeasibilityResult(
            bounds=result.bounds,
            current_storage_m3=(
                result.current_storage_m3
            ),
            inflow_m3s=(
                result.inflow_m3s
            ),
            timestep_seconds=(
                result.timestep_seconds
            ),
            low_storage_m3=(
                result.low_storage_m3
            ),
            high_storage_m3=(
                result.high_storage_m3
            ),
            safe_min_release_m3s=(
                safe_min_release
            ),
            safe_max_release_m3s=(
                result.safe_max_release_m3s
            ),
            candidate_releases_m3s=(
                candidate_releases
            ),
            feasible_mask=(
                feasible_mask
            ),
        )

    def _build_context(
        self,
        *,
        state: ReservoirSafetyState,
        use_p2: bool,
        p3_factor: Optional[float],
    ) -> ConstraintContext:
        p3 = None

        if p3_factor is not None:
            p3 = (
                self._build_p3_constraints(
                    state=state,
                    factor=p3_factor,
                )
            )

        return ConstraintContext(
            p1=state.p1,
            p2=(
                state.p2
                if use_p2
                else None
            ),
            p3=p3,
        )

    def _build_p3_constraints(
        self,
        *,
        state: ReservoirSafetyState,
        factor: float,
    ) -> P3Constraints:
        """Construct current P3 after the requested relaxation."""

        factor = _positive_finite(
            factor,
            "p3_factor",
        )

        if (
            factor
            < 1.0
            or factor
            > (
                self
                .max_p3_relaxation_factor
                + 1e-12
            )
        ):
            raise ValueError(
                "p3_factor is outside the "
                "configured relaxation range"
            )

        current_level = float(
            state.physics.level_from_storage(
                state.storage_m3
            )
        )

        level_change_limit = (
            self.level_change_limit_m[
                state.reservoir_id
            ]
            * factor
        )

        level_constraint = (
            LevelBounds(
                min_level_m=(
                    current_level
                    - level_change_limit
                ),
                max_level_m=(
                    current_level
                    + level_change_limit
                ),
            )
        )

        # First simulation day:
        #
        # previous_release_m3s is unavailable.
        #
        # Only the P3 release-change sub-constraint is disabled.
        # The P3 level-change constraint above remains active.
        if (
            state.previous_release_m3s
            is None
        ):
            release_constraint = None

        else:
            previous_release = float(
                state.previous_release_m3s
            )

            normal_change_limit = max(
                (
                    self.release_change_ratio
                    * previous_release
                ),
                self.release_change_floor_m3s,
            )

            release_change_limit = (
                normal_change_limit
                * factor
            )

            release_constraint = (
                ReleaseBounds(
                    min_release_m3s=max(
                        0.0,
                        (
                            previous_release
                            - release_change_limit
                        ),
                    ),
                    max_release_m3s=(
                        previous_release
                        + release_change_limit
                    ),
                )
            )

        return P3Constraints(
            level=level_constraint,
            release=release_constraint,
        )

    def _solve_p2_fallback(
        self,
        states,
    ):
        """Choose the P1-safe joint action with minimum P2 violation."""

        # Build the entire cascade graph again with P1 only.
        #
        # This deliberately still uses the official S201/S202/S203
        # implementations instead of reproducing another feasibility
        # calculation inside S3.
        p1_evaluation = (
            self._evaluate_s2(
                states=states,
                use_p2=False,
                p3_factor=None,
                p4_ratio=None,
            )
        )

        if not (
            p1_evaluation
            .extendability
            .has_joint_feasible_action
        ):
            raise RuntimeError(
                "No P1-safe joint action exists. "
                "P1 is an absolute physical "
                "safety constraint and cannot "
                "be relaxed."
            )

        root_state = states[
            0
        ]

        root_cost = (
            self._p2_violation_vector(
                state=root_state,
                inflow_m3s=(
                    root_state
                    .forcing_inflow_m3s
                ),
                feasible_mask=(
                    p1_evaluation
                    .wdd_local_result
                    .feasible_mask
                ),
            )
        )

        # Dynamic programming state:
        #
        # dynamic_cost[a_i]
        # =
        # minimum accumulated P2 violation of all partial paths
        # ending at action a_i.
        dynamic_cost = (
            root_cost.copy()
        )

        backpointers = []

        for link_index, compatibility in enumerate(
            p1_evaluation
            .compatibility_results
        ):
            downstream_state = states[
                link_index + 1
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
                compatible_mask = (
                    matrix[
                        upstream_action,
                        :
                    ]
                )

                if not np.any(
                    compatible_mask
                ):
                    continue

                candidate_inflow = float(
                    compatibility
                    .candidate_inflows_m3s[
                        upstream_action
                    ]
                )

                downstream_cost = (
                    self._p2_violation_vector(
                        state=(
                            downstream_state
                        ),
                        inflow_m3s=(
                            candidate_inflow
                        ),
                        feasible_mask=(
                            compatible_mask
                        ),
                    )
                )

                edge_cost[
                    upstream_action,
                    :
                ] = downstream_cost

            total_cost = (
                dynamic_cost[
                    :,
                    np.newaxis,
                ]
                + edge_cost
            )

            predecessor = np.argmin(
                total_cost,
                axis=0,
            )

            next_cost = total_cost[
                predecessor,
                np.arange(
                    total_cost.shape[1]
                ),
            ]

            invalid = ~np.isfinite(
                next_cost
            )

            predecessor = (
                predecessor.astype(
                    np.int64
                )
            )

            predecessor[
                invalid
            ] = -1

            backpointers.append(
                predecessor
            )

            dynamic_cost = (
                next_cost
            )

        final_action = int(
            np.argmin(
                dynamic_cost
            )
        )

        minimum_cost = float(
            dynamic_cost[
                final_action
            ]
        )

        if not math.isfinite(
            minimum_cost
        ):
            raise RuntimeError(
                "P1-safe S203 graph is nonempty, "
                "but P2 fallback could not "
                "construct a finite-cost path"
            )

        joint_action = [
            -1
            for _ in range(
                _NUM_RESERVOIRS
            )
        ]

        joint_action[
            -1
        ] = final_action

        for reservoir_index in range(
            _NUM_RESERVOIRS - 1,
            0,
            -1,
        ):
            predecessor = int(
                backpointers[
                    reservoir_index - 1
                ][
                    joint_action[
                        reservoir_index
                    ]
                ]
            )

            if predecessor < 0:
                raise RuntimeError(
                    "invalid P2 fallback "
                    "backpointer"
                )

            joint_action[
                reservoir_index - 1
            ] = predecessor

        return (
            tuple(
                joint_action
            ),
            minimum_cost,
            p1_evaluation,
        )

    def _p2_violation_vector(
        self,
        *,
        state: ReservoirSafetyState,
        inflow_m3s: float,
        feasible_mask,
    ) -> np.ndarray:
        """Compute normalized P2 violation for each P1-safe action."""

        releases = np.asarray(
            state.mapper
            .candidate_releases_m3s,
            dtype=np.float64,
        )

        feasible_mask = np.asarray(
            feasible_mask,
            dtype=bool,
        )

        if feasible_mask.shape != (
            releases.shape
        ):
            raise ValueError(
                f"{state.reservoir_id}: "
                "P2 cost feasible mask has "
                "invalid shape"
            )

        costs = np.full(
            releases.shape,
            np.inf,
            dtype=np.float64,
        )

        feasible_indices = (
            np.flatnonzero(
                feasible_mask
            )
        )

        if feasible_indices.size == 0:
            return costs

        timestep_seconds = (
            _positive_finite(
                state.physics
                .timestep_seconds,
                (
                    f"{state.reservoir_id}."
                    "timestep_seconds"
                ),
            )
        )

        next_storage = (
            state.storage_m3
            + (
                float(
                    inflow_m3s
                )
                - releases[
                    feasible_indices
                ]
            )
            * timestep_seconds
        )

        p1_min_level = float(
            state.p1.level.min_level_m
        )

        p1_max_level = float(
            state.p1.level.max_level_m
        )

        p1_min_storage = float(
            state.physics.storage_from_level(
                p1_min_level
            )
        )

        p1_max_storage = float(
            state.physics.storage_from_level(
                p1_max_level
            )
        )

        # P1 feasibility has already been checked by S201.
        # Clipping only protects the strict interpolation routine
        # against tiny floating-point boundary excursions.
        next_storage = np.clip(
            next_storage,
            p1_min_storage,
            p1_max_storage,
        )

        next_levels = (
            self._levels_from_storage(
                state.physics,
                next_storage,
            )
        )

        p2_min_level = (
            state.p2.level.min_level_m
        )

        p2_max_level = (
            state.p2.level.max_level_m
        )

        violation = np.zeros(
            next_levels.shape,
            dtype=np.float64,
        )

        if p2_max_level is not None:
            p2_max_level = float(
                p2_max_level
            )

            upper_mask = (
                next_levels
                > (
                    p2_max_level
                    + _LEVEL_TOLERANCE_M
                )
            )

            if np.any(
                upper_mask
            ):
                denominator = (
                    p1_max_level
                    - p2_max_level
                )

                if denominator <= 0.0:
                    violation[
                        upper_mask
                    ] = np.inf

                else:
                    violation[
                        upper_mask
                    ] += (
                        (
                            next_levels[
                                upper_mask
                            ]
                            - p2_max_level
                        )
                        / denominator
                    )

        if p2_min_level is not None:
            p2_min_level = float(
                p2_min_level
            )

            lower_mask = (
                next_levels
                < (
                    p2_min_level
                    - _LEVEL_TOLERANCE_M
                )
            )

            if np.any(
                lower_mask
            ):
                denominator = (
                    p2_min_level
                    - p1_min_level
                )

                if denominator <= 0.0:
                    violation[
                        lower_mask
                    ] = np.inf

                else:
                    violation[
                        lower_mask
                    ] += (
                        (
                            p2_min_level
                            - next_levels[
                                lower_mask
                            ]
                        )
                        / denominator
                    )

        costs[
            feasible_indices
        ] = violation

        return costs

    def _validate_states(
        self,
        states,
    ) -> None:
        if len(
            states
        ) != _NUM_RESERVOIRS:
            raise ValueError(
                "exactly five reservoir "
                "states are required"
            )

        for state in states:
            if not isinstance(
                state,
                ReservoirSafetyState,
            ):
                raise TypeError(
                    "states must contain "
                    "ReservoirSafetyState objects"
                )

        state_order = tuple(
            state.reservoir_id
            for state in states
        )

        if (
            state_order
            != self.reservoir_order
        ):
            raise ValueError(
                "states must follow "
                "WDD, BHT, XLD, XJB, THR"
            )

        for state in states:
            reservoir_id = (
                state.reservoir_id
            )

            if (
                state.mapper.num_actions
                < 2
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "at least two actions "
                    "are required"
                )

            p1_min_level = float(
                state.p1.level.min_level_m
            )

            p1_max_level = float(
                state.p1.level.max_level_m
            )

            current_level = float(
                state.physics.level_from_storage(
                    state.storage_m3
                )
            )

            if (
                current_level
                < (
                    p1_min_level
                    - _LEVEL_TOLERANCE_M
                )
                or current_level
                > (
                    p1_max_level
                    + _LEVEL_TOLERANCE_M
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "current state violates P1"
                )

            p2_min = (
                state.p2.level.min_level_m
            )

            p2_max = (
                state.p2.level.max_level_m
            )

            if (
                p2_min is not None
                and float(
                    p2_min
                )
                < (
                    p1_min_level
                    - _LEVEL_TOLERANCE_M
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "P2 minimum level lies "
                    "outside P1"
                )

            if (
                p2_max is not None
                and float(
                    p2_max
                )
                > (
                    p1_max_level
                    + _LEVEL_TOLERANCE_M
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "P2 maximum level lies "
                    "outside P1"
                )

            if (
                state.previous_release_m3s
                is not None
            ):
                previous_release = float(
                    state.previous_release_m3s
                )

                p1_min_release = float(
                    state.p1.release
                    .min_release_m3s
                )

                p1_max_release = float(
                    state.p1.release
                    .max_release_m3s
                )

                if (
                    previous_release
                    < (
                        p1_min_release
                        - _FLOW_TOLERANCE_M3S
                    )
                    or previous_release
                    > (
                        p1_max_release
                        + _FLOW_TOLERANCE_M3S
                    )
                ):
                    raise ValueError(
                        f"{reservoir_id}: "
                        "previous release lies "
                        "outside P1 release bounds"
                    )

    def _p3_factor_from_fraction(
        self,
        fraction: float,
    ) -> float:
        fraction = _finite(
            fraction,
            "P3 relaxation fraction",
        )

        if not (
            0.0
            <= fraction
            <= 1.0
        ):
            raise ValueError(
                "P3 relaxation fraction "
                "must be in [0, 1]"
            )

        return float(
            1.0
            + fraction
            * (
                self
                .max_p3_relaxation_factor
                - 1.0
            )
        )

    @staticmethod
    def _levels_from_storage(
        physics,
        storage_values,
    ) -> np.ndarray:
        values = np.asarray(
            storage_values,
            dtype=np.float64,
        )

        try:
            result = np.asarray(
                physics.level_from_storage(
                    values
                ),
                dtype=np.float64,
            )

            if result.shape == values.shape:
                return result

        except (
            TypeError,
            ValueError,
        ):
            pass

        return np.asarray(
            [
                float(
                    physics.level_from_storage(
                        value
                    )
                )
                for value in values
            ],
            dtype=np.float64,
        )


def _validate_num_actions(
    value,
    name: str,
) -> int:
    if (
        not isinstance(
            value,
            Integral,
        )
        or isinstance(
            value,
            (bool, np.bool_),
        )
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    value = int(
        value
    )

    if value < 2:
        raise ValueError(
            f"{name} must be at least 2"
        )

    return value


def _validate_action_index(
    *,
    action_index,
    num_actions: int,
    name: str,
) -> int:
    if (
        not isinstance(
            action_index,
            Integral,
        )
        or isinstance(
            action_index,
            (bool, np.bool_),
        )
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    action_index = int(
        action_index
    )

    if not (
        0
        <= action_index
        < num_actions
    ):
        raise IndexError(
            f"{name} must be in "
            f"[0, {num_actions - 1}]"
        )

    return action_index


def _finite(
    value,
    name: str,
) -> float:
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


def _nonnegative_finite(
    value,
    name: str,
) -> float:
    value = _finite(
        value,
        name,
    )

    if value < 0.0:
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return value


def _positive_finite(
    value,
    name: str,
) -> float:
    value = _finite(
        value,
        name,
    )

    if value <= 0.0:
        raise ValueError(
            f"{name} must be positive"
        )

    return value