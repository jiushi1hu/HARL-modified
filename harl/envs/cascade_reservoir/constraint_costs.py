from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

ECO_RELEASE = "eco_release"
GUARANTEED_POWER = "guaranteed_power"
NAVIGATION_RELEASE = "navigation_release"

_VALID_CONSTRAINT_TYPES = {
    ECO_RELEASE,
    GUARANTEED_POWER,
    NAVIGATION_RELEASE,
}


@dataclass(frozen=True)
class ConstraintSpec:
    reservoir_id: str
    constraint_type: str
    requirement: float
    scale_coefficient: float = 1.0
    normalizer: float = 1.0
    tolerance: float = 0.0
    budget: float = 0.0
    active_stages: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        reservoir_id = str(
            self.reservoir_id
        )

        constraint_type = str(
            self.constraint_type
        )

        if reservoir_id not in _RESERVOIR_ORDER:
            raise ValueError(
                f"unknown reservoir_id: "
                f"{reservoir_id}"
            )

        if (
            constraint_type
            not in _VALID_CONSTRAINT_TYPES
        ):
            raise ValueError(
                f"unknown constraint_type: "
                f"{constraint_type}"
            )

        requirement = _validate_positive(
            self.requirement,
            "requirement",
        )

        scale_coefficient = _validate_positive(
            self.scale_coefficient,
            "scale_coefficient",
        )

        if scale_coefficient != 1.0:
            raise ValueError("beta/scale_coefficient must equal 1 under the revised raw-violation formula")

        normalizer = _validate_positive(
            self.normalizer,
            "normalizer",
        )

        tolerance = _validate_nonnegative(
            self.tolerance,
            "tolerance",
        )

        budget = _validate_nonnegative(
            self.budget,
            "budget",
        )

        active_stages = self.active_stages

        if active_stages is not None:
            active_stages = tuple(
                str(stage)
                for stage
                in active_stages
            )

            if len(active_stages) == 0:
                raise ValueError(
                    "active_stages cannot "
                    "be empty"
                )

            if len(set(active_stages)) != len(
                active_stages
            ):
                raise ValueError(
                    "active_stages cannot "
                    "contain duplicates"
                )

        object.__setattr__(
            self,
            "reservoir_id",
            reservoir_id,
        )

        object.__setattr__(
            self,
            "constraint_type",
            constraint_type,
        )

        object.__setattr__(
            self,
            "requirement",
            requirement,
        )

        object.__setattr__(
            self,
            "scale_coefficient",
            scale_coefficient,
        )

        object.__setattr__(
            self,
            "normalizer",
            normalizer,
        )

        object.__setattr__(
            self,
            "tolerance",
            tolerance,
        )

        object.__setattr__(
            self,
            "budget",
            budget,
        )

        object.__setattr__(
            self,
            "active_stages",
            active_stages,
        )

    @property
    def constraint_id(self) -> str:
        return (
            f"{self.reservoir_id}:"
            f"{self.constraint_type}"
        )

    def is_active(
        self,
        operation_stage: str,
    ) -> bool:
        if self.active_stages is None:
            return True

        return (
            operation_stage
            in self.active_stages
        )


@dataclass(frozen=True)
class ConstraintCostResult:
    constraint_ids: Tuple[str, ...]
    raw_violations: np.ndarray
    normalized_violations: np.ndarray
    costs: np.ndarray
    active_flags: np.ndarray
    budgets: np.ndarray

    @property
    def num_constraints(self) -> int:
        return len(
            self.constraint_ids
        )

    def get_index(
        self,
        constraint_id: str,
    ) -> int:
        try:
            return self.constraint_ids.index(
                constraint_id
            )
        except ValueError as exc:
            raise KeyError(
                f"unknown constraint_id: "
                f"{constraint_id}"
            ) from exc

    def get_cost(
        self,
        constraint_id: str,
    ) -> float:
        index = self.get_index(
            constraint_id
        )

        return float(
            self.costs[
                index
            ]
        )

    def is_active(
        self,
        constraint_id: str,
    ) -> bool:
        index = self.get_index(
            constraint_id
        )

        return bool(
            self.active_flags[
                index
            ]
        )


class ConstraintCostEvaluator:

    def __init__(
        self,
        specs: Sequence[
            ConstraintSpec
        ],
    ):
        self.specs = tuple(
            specs
        )

        if len(
            self.specs
        ) == 0:
            raise ValueError(
                "at least one long-term "
                "constraint is required"
            )

        seen = set()

        for spec in self.specs:
            if not isinstance(
                spec,
                ConstraintSpec,
            ):
                raise TypeError(
                    "specs must contain "
                    "ConstraintSpec objects"
                )

            key = (
                spec.reservoir_id,
                spec.constraint_type,
            )

            if key in seen:
                raise ValueError(
                    "duplicate reservoir-constraint "
                    f"pair: {key}"
                )

            seen.add(
                key
            )

        self.constraint_ids = tuple(
            spec.constraint_id
            for spec
            in self.specs
        )

        self.budgets = np.asarray(
            [
                spec.budget
                for spec
                in self.specs
            ],
            dtype=np.float32,
        )

        self.budgets.setflags(
            write=False
        )

    @classmethod
    def from_config(
        cls,
        env_args: Mapping,
    ) -> "ConstraintCostEvaluator":

        config = env_args.get(
            "long_term_constraints"
        )

        if not isinstance(
            config,
            Sequence,
        ) or isinstance(
            config,
            (
                str,
                bytes,
            ),
        ):
            raise TypeError(
                "long_term_constraints must "
                "be a sequence"
            )

        specs = []

        for index, item in enumerate(
            config
        ):
            if not isinstance(
                item,
                Mapping,
            ):
                raise TypeError(
                    "each long-term constraint "
                    "must be a mapping"
                )

            required_keys = {
                "reservoir_id",
                "type",
                "requirement",
            }

            missing = (
                required_keys
                - set(
                    item
                )
            )

            if missing:
                raise KeyError(
                    f"constraint {index} missing "
                    f"keys: {missing}"
                )

            active_stages = item.get(
                "active_stages"
            )

            if active_stages is not None:
                active_stages = tuple(
                    active_stages
                )

            specs.append(
                ConstraintSpec(
                    reservoir_id=(
                        item[
                            "reservoir_id"
                        ]
                    ),
                    constraint_type=(
                        item[
                            "type"
                        ]
                    ),
                    requirement=(
                        item[
                            "requirement"
                        ]
                    ),
                    scale_coefficient=(
                        item.get(
                            "beta",
                            1.0,
                        )
                    ),
                    normalizer=(
                        item.get(
                            "normalizer",
                            1.0,
                        )
                    ),
                    tolerance=(
                        item.get(
                            "tolerance",
                            0.0,
                        )
                    ),
                    budget=(
                        item.get(
                            "budget",
                            0.0,
                        )
                    ),
                    active_stages=(
                        active_stages
                    ),
                )
            )

        return cls(
            specs
        )

    def evaluate(
        self,
        executed_releases_m3s: Mapping[
            str,
            float,
        ],
        powers_mw: Mapping[
            str,
            float,
        ],
        operation_stage: str,
    ) -> ConstraintCostResult:

        operation_stage = str(
            operation_stage
        )

        raw_violations = np.zeros(
            len(self.specs),
            dtype=np.float32,
        )

        normalized_violations = np.zeros(
            len(self.specs),
            dtype=np.float32,
        )

        costs = np.zeros(
            len(self.specs),
            dtype=np.float32,
        )

        active_flags = np.zeros(
            len(self.specs),
            dtype=np.float32,
        )

        for index, spec in enumerate(
            self.specs
        ):
            active = spec.is_active(
                operation_stage
            )

            active_flags[index] = float(active)

            actual_value = (
                self._get_actual_value(
                    spec=spec,
                    executed_releases_m3s=(
                        executed_releases_m3s
                    ),
                    powers_mw=(
                        powers_mw
                    ),
                )
            )

            raw_violation = (
                self._compute_raw_violation(
                    requirement=(
                        spec.requirement
                    ),
                    actual_value=(
                        actual_value
                    ),
                )
            )

            normalized_violation = (
                raw_violation
                / spec.normalizer
            )

            cost = max(
                0.0,
                normalized_violation
                - spec.tolerance,
            )

            raw_violations[
                index
            ] = raw_violation

            normalized_violations[
                index
            ] = (
                normalized_violation
            )

            costs[
                index
            ] = float(active) * cost

        raw_violations.setflags(
            write=False
        )

        normalized_violations.setflags(
            write=False
        )

        costs.setflags(
            write=False
        )

        active_flags.setflags(
            write=False
        )

        return ConstraintCostResult(
            constraint_ids=(
                self.constraint_ids
            ),
            raw_violations=(
                raw_violations
            ),
            normalized_violations=(
                normalized_violations
            ),
            costs=costs,
            active_flags=(
                active_flags
            ),
            budgets=self.budgets,
        )

    @staticmethod
    def _get_actual_value(
        spec: ConstraintSpec,
        executed_releases_m3s,
        powers_mw,
    ) -> float:

        reservoir_id = (
            spec.reservoir_id
        )

        if spec.constraint_type in (
            ECO_RELEASE,
            NAVIGATION_RELEASE,
        ):
            if (
                reservoir_id
                not in executed_releases_m3s
            ):
                raise KeyError(
                    f"missing executed release "
                    f"for {reservoir_id}"
                )

            value = float(
                executed_releases_m3s[
                    reservoir_id
                ]
            )

            return _validate_nonnegative(
                value,
                (
                    f"{reservoir_id} "
                    "executed_release"
                ),
            )

        if (
            spec.constraint_type
            == GUARANTEED_POWER
        ):
            if (
                reservoir_id
                not in powers_mw
            ):
                raise KeyError(
                    f"missing power "
                    f"for {reservoir_id}"
                )

            value = float(
                powers_mw[
                    reservoir_id
                ]
            )

            return _validate_nonnegative(
                value,
                (
                    f"{reservoir_id} "
                    "power_mw"
                ),
            )

        raise RuntimeError(
            "unsupported constraint type"
        )

    @staticmethod
    def _compute_raw_violation(
        requirement: float,
        actual_value: float,
    ) -> float:

        relative_shortfall = max(
            0.0,
            (
                requirement
                - actual_value
            )
            / requirement,
        )

        return float(
            math_sqrt(
                relative_shortfall
            )
        )


def math_sqrt(
    value: float,
) -> float:
    return float(
        np.sqrt(
            float(value)
        )
    )


def _validate_positive(
    value: float,
    name: str,
) -> float:
    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if value <= 0.0:
        raise ValueError(
            f"{name} must be positive"
        )

    return value


def _validate_nonnegative(
    value: float,
    name: str,
) -> float:
    value = float(
        value
    )

    if not np.isfinite(
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
