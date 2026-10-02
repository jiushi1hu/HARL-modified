from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Tuple

import numpy as np

from harl.envs.cascade_reservoir.action_mapping import (
    ReleaseActionMapper,
)
from harl.envs.cascade_reservoir.routing import (
    compute_downstream_inflow,
)
from harl.envs.cascade_reservoir.safety.s201_local_feasibility import (
    LocalFeasibilityResult,
)


_INFLOW_TOLERANCE_M3S = 1e-9


LocalFeasibilityEvaluator = Callable[
    [float],
    LocalFeasibilityResult,
]


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of S202 adjacent-reservoir compatibility evaluation."""

    upstream_reservoir_id: str
    downstream_reservoir_id: str

    # One candidate downstream inflow for each upstream action.
    candidate_inflows_m3s: np.ndarray

    # Shape:
    # (
    #     num_upstream_actions,
    #     num_downstream_actions,
    # )
    #
    # Rows    -> upstream candidate actions
    # Columns -> downstream candidate actions
    compatibility_matrix: np.ndarray

    def __post_init__(self):
        upstream_id = str(
            self.upstream_reservoir_id
        )
        downstream_id = str(
            self.downstream_reservoir_id
        )

        if not upstream_id:
            raise ValueError(
                "upstream_reservoir_id cannot be empty"
            )

        if not downstream_id:
            raise ValueError(
                "downstream_reservoir_id cannot be empty"
            )

        if upstream_id == downstream_id:
            raise ValueError(
                "upstream and downstream reservoirs "
                "must be different"
            )

        candidate_inflows = np.asarray(
            self.candidate_inflows_m3s,
            dtype=np.float64,
        )

        compatibility_matrix = np.asarray(
            self.compatibility_matrix,
            dtype=bool,
        )

        if candidate_inflows.ndim != 1:
            raise ValueError(
                "candidate_inflows_m3s must be "
                "one-dimensional"
            )

        if compatibility_matrix.ndim != 2:
            raise ValueError(
                "compatibility_matrix must be "
                "two-dimensional"
            )

        if candidate_inflows.size < 2:
            raise ValueError(
                "at least two upstream candidate "
                "actions are required"
            )

        if (
            compatibility_matrix.shape[0]
            != candidate_inflows.size
        ):
            raise ValueError(
                "compatibility_matrix row count must "
                "match candidate_inflows_m3s length"
            )

        if compatibility_matrix.shape[1] < 2:
            raise ValueError(
                "at least two downstream candidate "
                "actions are required"
            )

        if not np.all(
            np.isfinite(
                candidate_inflows
            )
        ):
            raise ValueError(
                "candidate_inflows_m3s contains "
                "non-finite values"
            )

        if np.any(
            candidate_inflows < 0.0
        ):
            raise ValueError(
                "candidate_inflows_m3s cannot "
                "contain negative values"
            )

        # Under the current daily no-routing-delay model:
        #
        # I_down_candidate
        # =
        # Q_up_target
        # +
        # I_interval
        #
        # Since the upstream action-to-release mapping is strictly
        # increasing, candidate downstream inflows must also be
        # strictly increasing.
        if not np.all(
            np.diff(
                candidate_inflows
            )
            > 0.0
        ):
            raise ValueError(
                "candidate_inflows_m3s must be "
                "strictly increasing"
            )

        candidate_inflows = (
            candidate_inflows.copy()
        )

        compatibility_matrix = (
            compatibility_matrix.copy()
        )

        candidate_inflows.setflags(
            write=False
        )

        compatibility_matrix.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "upstream_reservoir_id",
            upstream_id,
        )

        object.__setattr__(
            self,
            "downstream_reservoir_id",
            downstream_id,
        )

        object.__setattr__(
            self,
            "candidate_inflows_m3s",
            candidate_inflows,
        )

        object.__setattr__(
            self,
            "compatibility_matrix",
            compatibility_matrix,
        )

    @property
    def num_upstream_actions(
        self,
    ) -> int:
        return int(
            self.compatibility_matrix.shape[0]
        )

    @property
    def num_downstream_actions(
        self,
    ) -> int:
        return int(
            self.compatibility_matrix.shape[1]
        )

    @property
    def has_compatible_pair(
        self,
    ) -> bool:
        return bool(
            np.any(
                self.compatibility_matrix
            )
        )

    @property
    def num_compatible_pairs(
        self,
    ) -> int:
        return int(
            np.count_nonzero(
                self.compatibility_matrix
            )
        )

    def get_downstream_mask(
        self,
        upstream_action_index: int,
    ) -> np.ndarray:
        """Return downstream compatibility row for one upstream action."""

        upstream_action_index = (
            _validate_action_index(
                upstream_action_index,
                self.num_upstream_actions,
                "upstream_action_index",
            )
        )

        return self.compatibility_matrix[
            upstream_action_index,
            :,
        ].copy()

    def get_downstream_action_indices(
        self,
        upstream_action_index: int,
    ) -> Tuple[int, ...]:
        """Return compatible downstream action indices."""

        mask = self.get_downstream_mask(
            upstream_action_index
        )

        return tuple(
            int(index)
            for index
            in np.flatnonzero(
                mask
            )
        )


def build_compatibility_matrix(
    upstream_mapper: ReleaseActionMapper,
    downstream_mapper: ReleaseActionMapper,
    interval_inflow_m3s: float,
    evaluate_downstream: LocalFeasibilityEvaluator,
) -> CompatibilityResult:
    """Build the S202 binary compatibility matrix for one adjacent link.

    For upstream candidate action a_u:

        Q_up_target
        =
        upstream_mapper.action_to_release(a_u)

    the downstream candidate inflow is:

        I_down_candidate
        =
        Q_up_target
        +
        I_interval

    S201 is then evaluated once under this candidate inflow.
    Its full downstream feasible mask becomes one row of the
    compatibility matrix.

    S202 intentionally does not:
        - determine whether the upstream action itself is locally feasible;
        - perform S203 extendability recursion;
        - perform P2/P3/P4 relaxation;
        - use executed release Q_exec.

    Those responsibilities belong to other safety-layer stages.
    """

    if not isinstance(
        upstream_mapper,
        ReleaseActionMapper,
    ):
        raise TypeError(
            "upstream_mapper must be "
            "ReleaseActionMapper"
        )

    if not isinstance(
        downstream_mapper,
        ReleaseActionMapper,
    ):
        raise TypeError(
            "downstream_mapper must be "
            "ReleaseActionMapper"
        )

    if not callable(
        evaluate_downstream
    ):
        raise TypeError(
            "evaluate_downstream must be callable"
        )

    upstream_id = str(
        upstream_mapper.reservoir_id
    )

    downstream_id = str(
        downstream_mapper.reservoir_id
    )

    if upstream_id == downstream_id:
        raise ValueError(
            "upstream and downstream reservoirs "
            "must be different"
        )

    interval_inflow = (
        _nonnegative_finite(
            interval_inflow_m3s,
            "interval_inflow_m3s",
        )
    )

    upstream_releases = (
        _candidate_release_grid(
            upstream_mapper,
            "upstream_mapper",
        )
    )

    downstream_releases = (
        _candidate_release_grid(
            downstream_mapper,
            "downstream_mapper",
        )
    )

    num_upstream_actions = int(
        upstream_mapper.num_actions
    )

    num_downstream_actions = int(
        downstream_mapper.num_actions
    )

    candidate_inflows = np.empty(
        num_upstream_actions,
        dtype=np.float64,
    )

    compatibility_matrix = np.zeros(
        (
            num_upstream_actions,
            num_downstream_actions,
        ),
        dtype=bool,
    )

    # One S201 evaluation per upstream candidate action.
    #
    # S201 itself evaluates all downstream candidate releases
    # vectorially, so there is no need for a nested Python loop
    # over all downstream actions.
    for upstream_action_index in range(
        num_upstream_actions
    ):
        upstream_target_release = float(
            upstream_releases[
                upstream_action_index
            ]
        )

        # Important:
        #
        # S202 uses Q_target, not Q_exec.
        #
        # At this stage the upstream action has not actually been
        # executed.  We are only testing candidate compatibility.
        candidate_inflow = float(
            compute_downstream_inflow(
                upstream_flow=(
                    upstream_target_release
                ),
                interval_inflow=(
                    interval_inflow
                ),
            )
        )

        candidate_inflow = (
            _nonnegative_finite(
                candidate_inflow,
                "candidate_inflow_m3s",
            )
        )

        local_result = (
            evaluate_downstream(
                candidate_inflow
            )
        )

        _validate_local_result(
            local_result=local_result,
            expected_inflow_m3s=(
                candidate_inflow
            ),
            downstream_releases_m3s=(
                downstream_releases
            ),
            downstream_reservoir_id=(
                downstream_id
            ),
        )

        candidate_inflows[
            upstream_action_index
        ] = candidate_inflow

        # Use the authoritative boolean S201 result directly.
        #
        # Do not use action_mask here because action_mask is the
        # HARL-facing float32 representation.  S202 is an internal
        # logical feasibility calculation and should remain boolean.
        compatibility_matrix[
            upstream_action_index,
            :,
        ] = local_result.feasible_mask

    return CompatibilityResult(
        upstream_reservoir_id=(
            upstream_id
        ),
        downstream_reservoir_id=(
            downstream_id
        ),
        candidate_inflows_m3s=(
            candidate_inflows
        ),
        compatibility_matrix=(
            compatibility_matrix
        ),
    )


def _validate_local_result(
    *,
    local_result,
    expected_inflow_m3s: float,
    downstream_releases_m3s: np.ndarray,
    downstream_reservoir_id: str,
) -> None:
    """Validate that S201 evaluated exactly the requested downstream case."""

    if not isinstance(
        local_result,
        LocalFeasibilityResult,
    ):
        raise TypeError(
            "evaluate_downstream must return "
            "LocalFeasibilityResult"
        )

    # This catches an easy-to-miss closure/callback bug where an
    # evaluator ignores the candidate inflow passed by S202 and
    # accidentally reuses the reservoir's original forcing inflow.
    if not math.isclose(
        local_result.inflow_m3s,
        expected_inflow_m3s,
        rel_tol=0.0,
        abs_tol=_INFLOW_TOLERANCE_M3S,
    ):
        raise ValueError(
            f"{downstream_reservoir_id}: "
            "evaluate_downstream returned a result "
            "for a different inflow"
        )

    local_releases = np.asarray(
        local_result.candidate_releases_m3s,
        dtype=np.float64,
    )

    if local_releases.shape != (
        downstream_releases_m3s.shape
    ):
        raise ValueError(
            f"{downstream_reservoir_id}: "
            "S201 candidate release grid shape "
            "does not match downstream mapper"
        )

    # S201 and S202 must refer to exactly the same downstream
    # discrete action grid.  Shape equality alone is insufficient:
    # two reservoirs can have the same configured action count but different
    # Q_map_min / Q_map_max values.
    if not np.array_equal(
        local_releases,
        downstream_releases_m3s,
    ):
        raise ValueError(
            f"{downstream_reservoir_id}: "
            "S201 candidate release grid does not "
            "match downstream mapper"
        )

    local_mask = np.asarray(
        local_result.feasible_mask,
        dtype=bool,
    )

    if local_mask.shape != (
        downstream_releases_m3s.shape
    ):
        raise ValueError(
            f"{downstream_reservoir_id}: "
            "S201 feasible_mask shape does not "
            "match downstream mapper"
        )


def _candidate_release_grid(
    mapper: ReleaseActionMapper,
    name: str,
) -> np.ndarray:
    releases = np.asarray(
        mapper.candidate_releases_m3s,
        dtype=np.float64,
    )

    expected_shape = (
        int(
            mapper.num_actions
        ),
    )

    if releases.shape != expected_shape:
        raise ValueError(
            f"{name}.candidate_releases_m3s "
            f"must have shape {expected_shape}, "
            f"received {releases.shape}"
        )

    if not np.all(
        np.isfinite(
            releases
        )
    ):
        raise ValueError(
            f"{name}.candidate_releases_m3s "
            "contains non-finite values"
        )

    if np.any(
        releases < 0.0
    ):
        raise ValueError(
            f"{name}.candidate_releases_m3s "
            "cannot contain negative values"
        )

    if not np.all(
        np.diff(
            releases
        )
        > 0.0
    ):
        raise ValueError(
            f"{name}.candidate_releases_m3s "
            "must be strictly increasing"
        )

    return releases


def _validate_action_index(
    value,
    num_actions: int,
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

    if not (
        0
        <= value
        < num_actions
    ):
        raise IndexError(
            f"{name} must be in "
            f"[0, {num_actions - 1}]"
        )

    return value


def _nonnegative_finite(
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

    if value < 0.0:
        raise ValueError(
            f"{name} cannot be negative"
        )

    return value
