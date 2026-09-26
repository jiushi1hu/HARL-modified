from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from harl.envs.cascade_reservoir.action_mapping import (
    ReleaseActionMapper,
)
from harl.envs.cascade_reservoir.safety.constraint_context import (
    ConstraintContext,
    EffectiveS201Bounds,
)


_FLOW_TOLERANCE_M3S = 1e-7


@dataclass(frozen=True)
class LocalFeasibilityResult:
    """Result of S201 local candidate-action feasibility evaluation."""

    bounds: EffectiveS201Bounds

    current_storage_m3: float
    inflow_m3s: float
    timestep_seconds: float

    low_storage_m3: Optional[float]
    high_storage_m3: Optional[float]

    safe_min_release_m3s: Optional[float]
    safe_max_release_m3s: Optional[float]

    candidate_releases_m3s: np.ndarray
    feasible_mask: np.ndarray

    def __post_init__(self):
        if not isinstance(
            self.bounds,
            EffectiveS201Bounds,
        ):
            raise TypeError(
                "bounds must be EffectiveS201Bounds"
            )

        current_storage = _finite(
            self.current_storage_m3,
            "current_storage_m3",
        )

        inflow = _nonnegative_finite(
            self.inflow_m3s,
            "inflow_m3s",
        )

        timestep = _positive_finite(
            self.timestep_seconds,
            "timestep_seconds",
        )

        candidate_releases = np.asarray(
            self.candidate_releases_m3s,
            dtype=np.float64,
        )

        feasible_mask = np.asarray(
            self.feasible_mask,
            dtype=bool,
        )

        if candidate_releases.ndim != 1:
            raise ValueError(
                "candidate_releases_m3s must "
                "be one-dimensional"
            )

        if feasible_mask.shape != (
            candidate_releases.shape
        ):
            raise ValueError(
                "feasible_mask shape must match "
                "candidate release shape"
            )

        if candidate_releases.size < 2:
            raise ValueError(
                "at least two candidate actions "
                "are required"
            )

        if not np.all(
            np.isfinite(
                candidate_releases
            )
        ):
            raise ValueError(
                "candidate releases contain "
                "non-finite values"
            )

        if np.any(
            candidate_releases < 0.0
        ):
            raise ValueError(
                "candidate releases cannot "
                "be negative"
            )

        if not np.all(
            np.diff(
                candidate_releases
            )
            > 0.0
        ):
            raise ValueError(
                "candidate releases must be "
                "strictly increasing"
            )

        optional_values = (
            self.low_storage_m3,
            self.high_storage_m3,
            self.safe_min_release_m3s,
            self.safe_max_release_m3s,
        )

        if self.bounds.is_empty:
            if any(
                value is not None
                for value in optional_values
            ):
                raise ValueError(
                    "empty constraint bounds must "
                    "not contain derived S201 bounds"
                )

            if np.any(
                feasible_mask
            ):
                raise ValueError(
                    "empty constraint bounds cannot "
                    "have feasible actions"
                )

        else:
            if any(
                value is None
                for value in optional_values
            ):
                raise ValueError(
                    "non-empty constraint bounds "
                    "require complete S201 results"
                )

            low_storage = _finite(
                self.low_storage_m3,
                "low_storage_m3",
            )

            high_storage = _finite(
                self.high_storage_m3,
                "high_storage_m3",
            )

            safe_min = _finite(
                self.safe_min_release_m3s,
                "safe_min_release_m3s",
            )

            safe_max = _finite(
                self.safe_max_release_m3s,
                "safe_max_release_m3s",
            )

            if high_storage < low_storage:
                raise ValueError(
                    "high_storage_m3 cannot be "
                    "below low_storage_m3"
                )

            object.__setattr__(
                self,
                "low_storage_m3",
                low_storage,
            )

            object.__setattr__(
                self,
                "high_storage_m3",
                high_storage,
            )

            object.__setattr__(
                self,
                "safe_min_release_m3s",
                safe_min,
            )

            object.__setattr__(
                self,
                "safe_max_release_m3s",
                safe_max,
            )

        candidate_releases = (
            candidate_releases.copy()
        )

        feasible_mask = (
            feasible_mask.copy()
        )

        candidate_releases.setflags(
            write=False
        )

        feasible_mask.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "current_storage_m3",
            current_storage,
        )

        object.__setattr__(
            self,
            "inflow_m3s",
            inflow,
        )

        object.__setattr__(
            self,
            "timestep_seconds",
            timestep,
        )

        object.__setattr__(
            self,
            "candidate_releases_m3s",
            candidate_releases,
        )

        object.__setattr__(
            self,
            "feasible_mask",
            feasible_mask,
        )

    @property
    def num_actions(
        self,
    ) -> int:
        return int(
            self.candidate_releases_m3s.size
        )

    @property
    def has_feasible_action(
        self,
    ) -> bool:
        return bool(
            np.any(
                self.feasible_mask
            )
        )

    @property
    def num_feasible_actions(
        self,
    ) -> int:
        return int(
            np.count_nonzero(
                self.feasible_mask
            )
        )

    @property
    def feasible_action_indices(
        self,
    ) -> Tuple[int, ...]:
        return tuple(
            int(
                index
            )
            for index
            in np.flatnonzero(
                self.feasible_mask
            )
        )

    @property
    def action_mask(
        self,
    ) -> np.ndarray:
        """Return HARL-compatible 0/1 float action mask."""
        return self.feasible_mask.astype(
            np.float32
        )

    def is_action_feasible(
        self,
        action_index: int,
    ) -> bool:
        if isinstance(
            action_index,
            (bool, np.bool_),
        ):
            raise TypeError(
                "action_index cannot be boolean"
            )

        action_index = int(
            action_index
        )

        if not (
            0
            <= action_index
            < self.num_actions
        ):
            raise IndexError(
                "action_index out of range"
            )

        return bool(
            self.feasible_mask[
                action_index
            ]
        )


def evaluate_local_feasibility(
    *,
    physics,
    mapper: ReleaseActionMapper,
    storage_m3: float,
    inflow_m3s: float,
    constraints: Union[
        ConstraintContext,
        EffectiveS201Bounds,
    ],
    timestep_seconds: Optional[float] = None,
    flow_tolerance_m3s: float = (
        _FLOW_TOLERANCE_M3S
    ),
) -> LocalFeasibilityResult:
    """Evaluate S201 local feasibility for all candidate actions.

    For the current reservoir state and a given inflow, S201 computes

        Q_safe_min =
            max(
                Q_min,
                I + (V - V_high) / dt
            )

        Q_safe_max =
            min(
                Q_max,
                I + (V - V_low) / dt
            )

    and marks every discrete target release inside this interval as
    locally feasible.

    No constraint relaxation is performed here.
    """

    if not isinstance(
        mapper,
        ReleaseActionMapper,
    ):
        raise TypeError(
            "mapper must be ReleaseActionMapper"
        )

    current_storage = _finite(
        storage_m3,
        "storage_m3",
    )

    if current_storage < 0.0:
        raise ValueError(
            "storage_m3 cannot be negative"
        )

    inflow = _nonnegative_finite(
        inflow_m3s,
        "inflow_m3s",
    )

    tolerance = _nonnegative_finite(
        flow_tolerance_m3s,
        "flow_tolerance_m3s",
    )

    timestep = _resolve_timestep(
        physics=physics,
        timestep_seconds=(
            timestep_seconds
        ),
    )

    bounds = _resolve_bounds(
        constraints
    )

    candidate_releases = (
        _candidate_releases(
            mapper
        )
    )

    # Validate that the current storage itself belongs to the
    # known physical Z-V relation.  The return value is not needed;
    # level_from_storage() is called deliberately for strict range
    # validation.
    physics.level_from_storage(
        current_storage
    )

    # Critical ordering:
    #
    # P1/P2/P3 intersection may legitimately be empty.  In that
    # situation S201 must immediately return an all-false mask.
    #
    # Do NOT attempt storage_from_level(min/max) first, because an
    # empty intersection is a normal feasibility outcome rather than
    # an interpolation problem.
    if bounds.is_empty:
        return LocalFeasibilityResult(
            bounds=bounds,
            current_storage_m3=(
                current_storage
            ),
            inflow_m3s=inflow,
            timestep_seconds=timestep,
            low_storage_m3=None,
            high_storage_m3=None,
            safe_min_release_m3s=None,
            safe_max_release_m3s=None,
            candidate_releases_m3s=(
                candidate_releases
            ),
            feasible_mask=np.zeros(
                mapper.num_actions,
                dtype=bool,
            ),
        )

    low_storage = float(
        physics.storage_from_level(
            bounds.min_level_m
        )
    )

    high_storage = float(
        physics.storage_from_level(
            bounds.max_level_m
        )
    )

    if (
        not math.isfinite(
            low_storage
        )
        or not math.isfinite(
            high_storage
        )
    ):
        raise RuntimeError(
            "level-storage interpolation "
            "produced non-finite storage"
        )

    if high_storage < low_storage:
        raise RuntimeError(
            "effective water-level bounds "
            "produce reversed storage bounds"
        )

    safe_min_release = max(
        bounds.min_release_m3s,
        (
            inflow
            + (
                current_storage
                - high_storage
            )
            / timestep
        ),
    )

    safe_max_release = min(
        bounds.max_release_m3s,
        (
            inflow
            + (
                current_storage
                - low_storage
            )
            / timestep
        ),
    )

    safe_min_release = float(
        safe_min_release
    )

    safe_max_release = float(
        safe_max_release
    )

    if (
        not math.isfinite(
            safe_min_release
        )
        or not math.isfinite(
            safe_max_release
        )
    ):
        raise RuntimeError(
            "S201 produced non-finite "
            "release bounds"
        )

    if (
        safe_min_release
        > safe_max_release
        + tolerance
    ):
        feasible_mask = np.zeros(
            mapper.num_actions,
            dtype=bool,
        )

    else:
        feasible_mask = (
            candidate_releases
            >= (
                safe_min_release
                - tolerance
            )
        ) & (
            candidate_releases
            <= (
                safe_max_release
                + tolerance
            )
        )

    return LocalFeasibilityResult(
        bounds=bounds,
        current_storage_m3=(
            current_storage
        ),
        inflow_m3s=inflow,
        timestep_seconds=timestep,
        low_storage_m3=(
            low_storage
        ),
        high_storage_m3=(
            high_storage
        ),
        safe_min_release_m3s=(
            safe_min_release
        ),
        safe_max_release_m3s=(
            safe_max_release
        ),
        candidate_releases_m3s=(
            candidate_releases
        ),
        feasible_mask=(
            feasible_mask
        ),
    )


def _resolve_bounds(
    constraints,
) -> EffectiveS201Bounds:
    if isinstance(
        constraints,
        ConstraintContext,
    ):
        return constraints.resolve()

    if isinstance(
        constraints,
        EffectiveS201Bounds,
    ):
        return constraints

    raise TypeError(
        "constraints must be either "
        "ConstraintContext or "
        "EffectiveS201Bounds"
    )


def _candidate_releases(
    mapper: ReleaseActionMapper,
) -> np.ndarray:
    if hasattr(
        mapper,
        "candidate_releases_m3s",
    ):
        releases = getattr(
            mapper,
            "candidate_releases_m3s",
        )

        if callable(
            releases
        ):
            releases = releases()

        releases = np.asarray(
            releases,
            dtype=np.float64,
        )

    else:
        releases = np.asarray(
            [
                mapper.action_to_release(
                    action_index
                )
                for action_index
                in range(
                    mapper.num_actions
                )
            ],
            dtype=np.float64,
        )

    expected_shape = (
        mapper.num_actions,
    )

    if releases.shape != expected_shape:
        raise ValueError(
            "candidate release array must "
            f"have shape {expected_shape}, "
            f"received {releases.shape}"
        )

    if not np.all(
        np.isfinite(
            releases
        )
    ):
        raise ValueError(
            "candidate release array contains "
            "non-finite values"
        )

    if np.any(
        releases < 0.0
    ):
        raise ValueError(
            "candidate releases cannot "
            "be negative"
        )

    if not np.all(
        np.diff(
            releases
        )
        > 0.0
    ):
        raise ValueError(
            "candidate releases must be "
            "strictly increasing"
        )

    return releases


def _resolve_timestep(
    physics,
    timestep_seconds,
) -> float:
    if timestep_seconds is None:
        if not hasattr(
            physics,
            "timestep_seconds",
        ):
            raise AttributeError(
                "physics must expose "
                "timestep_seconds when no "
                "explicit timestep is supplied"
            )

        timestep_seconds = (
            physics.timestep_seconds
        )

    return _positive_finite(
        timestep_seconds,
        "timestep_seconds",
    )


def _finite(
    value,
    name,
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
    name,
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
    name,
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