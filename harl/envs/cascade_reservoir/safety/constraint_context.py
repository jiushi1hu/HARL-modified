from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LevelBounds:
    """Water-level bounds in metres.

    None means that the corresponding side does not impose an
    additional restriction.

    This low-level class intentionally allows an empty interval
    (min_level_m > max_level_m), because an empty interval may arise
    after intersecting individually valid P1/P2/P3 constraints.
    """

    min_level_m: Optional[float] = None
    max_level_m: Optional[float] = None

    def __post_init__(self):
        minimum = _optional_finite(
            self.min_level_m,
            "min_level_m",
        )

        maximum = _optional_finite(
            self.max_level_m,
            "max_level_m",
        )

        object.__setattr__(
            self,
            "min_level_m",
            minimum,
        )

        object.__setattr__(
            self,
            "max_level_m",
            maximum,
        )

    @property
    def is_complete(self) -> bool:
        return (
            self.min_level_m is not None
            and self.max_level_m is not None
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.min_level_m is not None
            and self.max_level_m is not None
            and self.min_level_m
            > self.max_level_m
        )

    def intersect(
        self,
        other: "LevelBounds",
    ) -> "LevelBounds":
        if not isinstance(
            other,
            LevelBounds,
        ):
            raise TypeError(
                "other must be LevelBounds"
            )

        minimum = _max_optional(
            self.min_level_m,
            other.min_level_m,
        )

        maximum = _min_optional(
            self.max_level_m,
            other.max_level_m,
        )

        return LevelBounds(
            min_level_m=minimum,
            max_level_m=maximum,
        )

    def contains(
        self,
        level_m: float,
        tolerance: float = 0.0,
    ) -> bool:
        value = _finite(
            level_m,
            "level_m",
        )

        tolerance = _nonnegative_finite(
            tolerance,
            "tolerance",
        )

        if self.is_empty:
            return False

        if (
            self.min_level_m is not None
            and value
            < self.min_level_m - tolerance
        ):
            return False

        if (
            self.max_level_m is not None
            and value
            > self.max_level_m + tolerance
        ):
            return False

        return True


@dataclass(frozen=True)
class ReleaseBounds:
    """Total-release bounds in m3/s.

    None means that the corresponding side does not impose an
    additional restriction.

    As with LevelBounds, an empty interval is allowed here because
    intersection of individually valid constraints may legitimately
    produce an empty feasible set.
    """

    min_release_m3s: Optional[float] = None
    max_release_m3s: Optional[float] = None

    def __post_init__(self):
        minimum = _optional_finite(
            self.min_release_m3s,
            "min_release_m3s",
        )

        maximum = _optional_finite(
            self.max_release_m3s,
            "max_release_m3s",
        )

        if (
            minimum is not None
            and minimum < 0.0
        ):
            raise ValueError(
                "min_release_m3s cannot "
                "be negative"
            )

        if (
            maximum is not None
            and maximum < 0.0
        ):
            raise ValueError(
                "max_release_m3s cannot "
                "be negative"
            )

        object.__setattr__(
            self,
            "min_release_m3s",
            minimum,
        )

        object.__setattr__(
            self,
            "max_release_m3s",
            maximum,
        )

    @property
    def is_complete(self) -> bool:
        return (
            self.min_release_m3s is not None
            and self.max_release_m3s is not None
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.min_release_m3s is not None
            and self.max_release_m3s is not None
            and self.min_release_m3s
            > self.max_release_m3s
        )

    def intersect(
        self,
        other: "ReleaseBounds",
    ) -> "ReleaseBounds":
        if not isinstance(
            other,
            ReleaseBounds,
        ):
            raise TypeError(
                "other must be ReleaseBounds"
            )

        minimum = _max_optional(
            self.min_release_m3s,
            other.min_release_m3s,
        )

        maximum = _min_optional(
            self.max_release_m3s,
            other.max_release_m3s,
        )

        return ReleaseBounds(
            min_release_m3s=minimum,
            max_release_m3s=maximum,
        )

    def contains(
        self,
        release_m3s: float,
        tolerance: float = 0.0,
    ) -> bool:
        value = _finite(
            release_m3s,
            "release_m3s",
        )

        tolerance = _nonnegative_finite(
            tolerance,
            "tolerance",
        )

        if value < 0.0:
            return False

        if self.is_empty:
            return False

        if (
            self.min_release_m3s is not None
            and value
            < self.min_release_m3s - tolerance
        ):
            return False

        if (
            self.max_release_m3s is not None
            and value
            > self.max_release_m3s + tolerance
        ):
            return False

        return True


@dataclass(frozen=True)
class P1Constraints:
    """Absolute physical safety constraints.

    P1 must always define complete, non-empty water-level and total
    release intervals.  P1 is never relaxed.
    """

    level: LevelBounds
    release: ReleaseBounds

    def __post_init__(self):
        if not isinstance(
            self.level,
            LevelBounds,
        ):
            raise TypeError(
                "P1 level must be LevelBounds"
            )

        if not isinstance(
            self.release,
            ReleaseBounds,
        ):
            raise TypeError(
                "P1 release must be "
                "ReleaseBounds"
            )

        if not self.level.is_complete:
            raise ValueError(
                "P1 must define both minimum "
                "and maximum water levels"
            )

        if self.level.is_empty:
            raise ValueError(
                "P1 water-level interval "
                "cannot be empty"
            )

        if not self.release.is_complete:
            raise ValueError(
                "P1 must define both minimum "
                "and maximum total releases"
            )

        if self.release.is_empty:
            raise ValueError(
                "P1 total-release interval "
                "cannot be empty"
            )


@dataclass(frozen=True)
class P2Constraints:
    """Stage-dependent operating water-level constraints.

    Either side may be None if that operating stage does not impose
    an additional lower or upper level restriction.
    """

    level: LevelBounds

    def __post_init__(self):
        if not isinstance(
            self.level,
            LevelBounds,
        ):
            raise TypeError(
                "P2 level must be LevelBounds"
            )

        if self.level.is_empty:
            raise ValueError(
                "P2 water-level interval "
                "cannot be intrinsically empty"
            )


@dataclass(frozen=True)
class P3Constraints:
    """Adjacent-period smooth-operation constraints.

    After a P3 change-rate rule is converted to absolute bounds, the
    resulting level and release intervals can be stored here.

    A None field means that the corresponding P3 sub-constraint is
    inactive.  This is important on the first simulation day:
    release may be None because no verified previous-day release is
    available, while the level-change constraint can remain active.
    """

    level: Optional[LevelBounds] = None
    release: Optional[ReleaseBounds] = None

    def __post_init__(self):
        if (
            self.level is not None
            and not isinstance(
                self.level,
                LevelBounds,
            )
        ):
            raise TypeError(
                "P3 level must be "
                "LevelBounds or None"
            )

        if (
            self.release is not None
            and not isinstance(
                self.release,
                ReleaseBounds,
            )
        ):
            raise TypeError(
                "P3 release must be "
                "ReleaseBounds or None"
            )

        if (
            self.level is not None
            and self.level.is_empty
        ):
            raise ValueError(
                "P3 water-level interval "
                "cannot be intrinsically empty"
            )

        if (
            self.release is not None
            and self.release.is_empty
        ):
            raise ValueError(
                "P3 release interval "
                "cannot be intrinsically empty"
            )


@dataclass(frozen=True)
class ConstraintContext:
    """Constraint set used by S201 local-feasibility evaluation."""

    p1: P1Constraints
    p2: Optional[P2Constraints] = None
    p3: Optional[P3Constraints] = None

    def __post_init__(self):
        if not isinstance(
            self.p1,
            P1Constraints,
        ):
            raise TypeError(
                "p1 must be P1Constraints"
            )

        if (
            self.p2 is not None
            and not isinstance(
                self.p2,
                P2Constraints,
            )
        ):
            raise TypeError(
                "p2 must be "
                "P2Constraints or None"
            )

        if (
            self.p3 is not None
            and not isinstance(
                self.p3,
                P3Constraints,
            )
        ):
            raise TypeError(
                "p3 must be "
                "P3Constraints or None"
            )

    def resolve(
        self,
    ) -> "EffectiveS201Bounds":
        return resolve_s201_bounds(
            self
        )


@dataclass(frozen=True)
class EffectiveS201Bounds:
    """Intersection of the currently active P1/P2/P3 constraints."""

    level: LevelBounds
    release: ReleaseBounds

    def __post_init__(self):
        if not isinstance(
            self.level,
            LevelBounds,
        ):
            raise TypeError(
                "level must be LevelBounds"
            )

        if not isinstance(
            self.release,
            ReleaseBounds,
        ):
            raise TypeError(
                "release must be "
                "ReleaseBounds"
            )

        # Because P1 is complete, an effective S201 interval must
        # remain numerically complete even if it is empty.
        if not self.level.is_complete:
            raise ValueError(
                "effective level bounds "
                "must be complete"
            )

        if not self.release.is_complete:
            raise ValueError(
                "effective release bounds "
                "must be complete"
            )

    @property
    def min_level_m(
        self,
    ) -> float:
        return float(
            self.level.min_level_m
        )

    @property
    def max_level_m(
        self,
    ) -> float:
        return float(
            self.level.max_level_m
        )

    @property
    def min_release_m3s(
        self,
    ) -> float:
        return float(
            self.release.min_release_m3s
        )

    @property
    def max_release_m3s(
        self,
    ) -> float:
        return float(
            self.release.max_release_m3s
        )

    @property
    def level_is_empty(
        self,
    ) -> bool:
        return self.level.is_empty

    @property
    def release_is_empty(
        self,
    ) -> bool:
        return self.release.is_empty

    @property
    def is_empty(
        self,
    ) -> bool:
        return (
            self.level.is_empty
            or self.release.is_empty
        )


def resolve_s201_bounds(
    context_or_p1,
    p2: Optional[P2Constraints] = None,
    p3: Optional[P3Constraints] = None,
) -> EffectiveS201Bounds:
    """Resolve the effective S201 bounds.

    Supported forms:

        resolve_s201_bounds(context)

    or:

        resolve_s201_bounds(
            p1,
            p2,
            p3,
        )

    The function only intersects currently active constraints.
    It never relaxes any constraint.
    """

    if isinstance(
        context_or_p1,
        ConstraintContext,
    ):
        if (
            p2 is not None
            or p3 is not None
        ):
            raise ValueError(
                "when ConstraintContext is supplied, "
                "p2 and p3 must not be supplied "
                "separately"
            )

        p1 = context_or_p1.p1
        p2 = context_or_p1.p2
        p3 = context_or_p1.p3

    elif isinstance(
        context_or_p1,
        P1Constraints,
    ):
        p1 = context_or_p1

        if (
            p2 is not None
            and not isinstance(
                p2,
                P2Constraints,
            )
        ):
            raise TypeError(
                "p2 must be "
                "P2Constraints or None"
            )

        if (
            p3 is not None
            and not isinstance(
                p3,
                P3Constraints,
            )
        ):
            raise TypeError(
                "p3 must be "
                "P3Constraints or None"
            )

    else:
        raise TypeError(
            "first argument must be either "
            "ConstraintContext or "
            "P1Constraints"
        )

    effective_level = (
        p1.level
    )

    effective_release = (
        p1.release
    )

    if p2 is not None:
        effective_level = (
            effective_level.intersect(
                p2.level
            )
        )

    if p3 is not None:
        if p3.level is not None:
            effective_level = (
                effective_level.intersect(
                    p3.level
                )
            )

        if p3.release is not None:
            effective_release = (
                effective_release.intersect(
                    p3.release
                )
            )

    return EffectiveS201Bounds(
        level=effective_level,
        release=effective_release,
    )


def _optional_finite(
    value,
    name,
):
    if value is None:
        return None

    return _finite(
        value,
        name,
    )


def _finite(
    value,
    name,
):
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
):
    value = _finite(
        value,
        name,
    )

    if value < 0.0:
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return value


def _max_optional(
    first,
    second,
):
    if first is None:
        return second

    if second is None:
        return first

    return max(
        first,
        second,
    )


def _min_optional(
    first,
    second,
):
    if first is None:
        return second

    if second is None:
        return first

    return min(
        first,
        second,
    )