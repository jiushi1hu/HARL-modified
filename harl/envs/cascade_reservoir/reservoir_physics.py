from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np


_WATER_DENSITY_KG_M3 = 1000.0
_GRAVITY_M_S2 = 9.81
_WATTS_PER_MW = 1_000_000.0
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class PhysicalTransition:
    next_storage_m3: float
    next_level_m: float
    turbine_release_m3s: float
    non_generation_release_m3s: float
    power_mw: float
    energy_mwh: float

    @property
    def total_release_m3s(self) -> float:
        return (
            self.turbine_release_m3s
            + self.non_generation_release_m3s
        )


class ReservoirPhysics:
    """Physical model of one reservoir.

    The current implementation uses:

    1. strict piecewise-linear level-storage interpolation;
    2. water balance without evaporation or other unmodelled losses;
    3. turbine-first release allocation;
    4. fixed representative net head;
    5. installed-capacity-limited hydropower production.

    The representative head is an engineering simplification of the
    current first runnable version.  It can later be replaced by a
    dynamic head model without changing the environment interface.
    """

    def __init__(
        self,
        reservoir_id: str,
        installed_capacity_mw: float,
        max_turbine_flow_m3s: float,
        max_total_release_m3s: float,
        typical_net_head_m: float,
        level_anchors_m,
        storage_anchors_m3,
        efficiency: float = 0.94,
        timestep_seconds: float = 86400.0,
    ):
        self.reservoir_id = str(
            reservoir_id
        )

        if not self.reservoir_id:
            raise ValueError(
                "reservoir_id cannot be empty"
            )

        self.installed_capacity_mw = (
            self._positive_number(
                installed_capacity_mw,
                "installed_capacity_mw",
            )
        )

        self.max_turbine_flow_m3s = (
            self._positive_number(
                max_turbine_flow_m3s,
                "max_turbine_flow_m3s",
            )
        )

        self.max_total_release_m3s = (
            self._positive_number(
                max_total_release_m3s,
                "max_total_release_m3s",
            )
        )

        if (
            self.max_turbine_flow_m3s
            > self.max_total_release_m3s
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "max_turbine_flow_m3s cannot "
                "exceed max_total_release_m3s"
            )

        self.typical_net_head_m = (
            self._positive_number(
                typical_net_head_m,
                "typical_net_head_m",
            )
        )

        self.efficiency = float(
            efficiency
        )

        if (
            not math.isfinite(
                self.efficiency
            )
            or not (
                0.0
                < self.efficiency
                <= 1.0
            )
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "efficiency must lie in (0, 1]"
            )

        self.timestep_seconds = (
            self._positive_number(
                timestep_seconds,
                "timestep_seconds",
            )
        )

        (
            self._level_anchors_m,
            self._storage_anchors_m3,
        ) = self._prepare_level_storage_curve(
            level_anchors_m,
            storage_anchors_m3,
        )

        self.min_level_m = float(
            self._level_anchors_m[
                0
            ]
        )

        self.max_level_m = float(
            self._level_anchors_m[
                -1
            ]
        )

        self.min_storage_m3 = float(
            self._storage_anchors_m3[
                0
            ]
        )

        self.max_storage_m3 = float(
            self._storage_anchors_m3[
                -1
            ]
        )

    @property
    def level_anchors_m(
        self,
    ) -> np.ndarray:
        result = (
            self._level_anchors_m.copy()
        )

        result.setflags(
            write=False
        )

        return result

    @property
    def storage_anchors_m3(
        self,
    ) -> np.ndarray:
        result = (
            self._storage_anchors_m3.copy()
        )

        result.setflags(
            write=False
        )

        return result

    def storage_from_level(
        self,
        level_m,
    ):
        """Convert water level to storage without extrapolation."""
        return self._strict_interp(
            values=level_m,
            x=self._level_anchors_m,
            y=self._storage_anchors_m3,
            value_name="level_m",
        )

    def level_from_storage(
        self,
        storage_m3,
    ):
        """Convert storage to water level without extrapolation."""
        return self._strict_interp(
            values=storage_m3,
            x=self._storage_anchors_m3,
            y=self._level_anchors_m,
            value_name="storage_m3",
        )

    # Compatibility aliases.  The project should use
    # storage_from_level() / level_from_storage() as the canonical API.
    def level_to_storage(
        self,
        level_m,
    ):
        return self.storage_from_level(
            level_m
        )

    def storage_to_level(
        self,
        storage_m3,
    ):
        return self.level_from_storage(
            storage_m3
        )

    def compute_next_storage(
        self,
        storage_m3: float,
        inflow_m3s: float,
        total_release_m3s: float,
        timestep_seconds: float = None,
    ) -> float:
        storage = self._validate_storage(
            storage_m3
        )

        inflow = self._nonnegative_number(
            inflow_m3s,
            "inflow_m3s",
        )

        release = self._validate_release(
            total_release_m3s
        )

        dt = self._resolve_timestep(
            timestep_seconds
        )

        next_storage = (
            storage
            + (
                inflow
                - release
            )
            * dt
        )

        if not math.isfinite(
            next_storage
        ):
            raise RuntimeError(
                f"{self.reservoir_id}: "
                "water balance produced "
                "non-finite storage"
            )

        return float(
            next_storage
        )

    def split_release(
        self,
        total_release_m3s: float,
    ):
        """Allocate total release to turbine flow first."""
        release = self._validate_release(
            total_release_m3s
        )

        turbine_release = min(
            release,
            self.max_turbine_flow_m3s,
        )

        non_generation_release = max(
            0.0,
            (
                release
                - turbine_release
            ),
        )

        return (
            float(
                turbine_release
            ),
            float(
                non_generation_release
            ),
        )

    def compute_power_mw(
        self,
        turbine_release_m3s: float,
    ) -> float:
        turbine_release = (
            self._nonnegative_number(
                turbine_release_m3s,
                "turbine_release_m3s",
            )
        )

        if (
            turbine_release
            > (
                self.max_turbine_flow_m3s
                + 1e-9
            )
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "turbine flow exceeds "
                "maximum turbine flow"
            )

        hydraulic_power_mw = (
            self.efficiency
            * _WATER_DENSITY_KG_M3
            * _GRAVITY_M_S2
            * self.typical_net_head_m
            * turbine_release
            / _WATTS_PER_MW
        )

        power_mw = min(
            self.installed_capacity_mw,
            hydraulic_power_mw,
        )

        if not math.isfinite(
            power_mw
        ):
            raise RuntimeError(
                f"{self.reservoir_id}: "
                "power calculation produced "
                "non-finite value"
            )

        return float(
            max(
                0.0,
                power_mw,
            )
        )

    def compute_energy_mwh(
        self,
        power_mw: float,
        timestep_seconds: float = None,
    ) -> float:
        power = self._nonnegative_number(
            power_mw,
            "power_mw",
        )

        dt = self._resolve_timestep(
            timestep_seconds
        )

        energy_mwh = (
            power
            * dt
            / _SECONDS_PER_HOUR
        )

        if not math.isfinite(
            energy_mwh
        ):
            raise RuntimeError(
                f"{self.reservoir_id}: "
                "energy calculation produced "
                "non-finite value"
            )

        return float(
            energy_mwh
        )

    def transition(
        self,
        storage_m3: float,
        inflow_m3s: float,
        total_release_m3s: float,
        timestep_seconds: float = None,
    ) -> PhysicalTransition:
        """Execute one physical reservoir transition."""

        dt = self._resolve_timestep(
            timestep_seconds
        )

        next_storage = (
            self.compute_next_storage(
                storage_m3=storage_m3,
                inflow_m3s=inflow_m3s,
                total_release_m3s=(
                    total_release_m3s
                ),
                timestep_seconds=dt,
            )
        )

        # Strictly forbid physical extrapolation of the
        # level-storage relation.  S201/S3 should prevent
        # substantive violations before transition().
        next_level = float(
            self.level_from_storage(
                next_storage
            )
        )

        (
            turbine_release,
            non_generation_release,
        ) = self.split_release(
            total_release_m3s
        )

        power_mw = (
            self.compute_power_mw(
                turbine_release
            )
        )

        energy_mwh = (
            self.compute_energy_mwh(
                power_mw=power_mw,
                timestep_seconds=dt,
            )
        )

        return PhysicalTransition(
            next_storage_m3=float(
                next_storage
            ),
            next_level_m=float(
                next_level
            ),
            turbine_release_m3s=float(
                turbine_release
            ),
            non_generation_release_m3s=float(
                non_generation_release
            ),
            power_mw=float(
                power_mw
            ),
            energy_mwh=float(
                energy_mwh
            ),
        )

    def _validate_storage(
        self,
        storage_m3: float,
    ) -> float:
        storage = float(
            storage_m3
        )

        if not math.isfinite(
            storage
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "storage_m3 must be finite"
            )

        if storage < 0.0:
            raise ValueError(
                f"{self.reservoir_id}: "
                "storage_m3 cannot be negative"
            )

        # Current storage must lie inside the known
        # level-storage curve.  No extrapolation is allowed.
        self._check_scalar_in_range(
            value=storage,
            lower=self.min_storage_m3,
            upper=self.max_storage_m3,
            name="storage_m3",
        )

        return storage

    def _validate_release(
        self,
        total_release_m3s: float,
    ) -> float:
        release = self._nonnegative_number(
            total_release_m3s,
            "total_release_m3s",
        )

        tolerance = max(
            1e-9,
            self.max_total_release_m3s
            * 1e-12,
        )

        if (
            release
            > (
                self.max_total_release_m3s
                + tolerance
            )
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "total release "
                f"{release} m3/s exceeds "
                "physical maximum "
                f"{self.max_total_release_m3s} m3/s"
            )

        if (
            release
            > self.max_total_release_m3s
        ):
            release = (
                self.max_total_release_m3s
            )

        return float(
            release
        )

    def _resolve_timestep(
        self,
        timestep_seconds,
    ) -> float:
        if timestep_seconds is None:
            return float(
                self.timestep_seconds
            )

        timestep = self._positive_number(
            timestep_seconds,
            "timestep_seconds",
        )

        return float(
            timestep
        )

    def _prepare_level_storage_curve(
        self,
        level_anchors_m,
        storage_anchors_m3,
    ):
        levels = np.asarray(
            level_anchors_m,
            dtype=np.float64,
        )

        storages = np.asarray(
            storage_anchors_m3,
            dtype=np.float64,
        )

        if levels.ndim != 1:
            raise ValueError(
                f"{self.reservoir_id}: "
                "level anchors must be "
                "one-dimensional"
            )

        if storages.ndim != 1:
            raise ValueError(
                f"{self.reservoir_id}: "
                "storage anchors must be "
                "one-dimensional"
            )

        if (
            levels.shape
            != storages.shape
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "level and storage anchor "
                "arrays must have equal length"
            )

        if levels.size < 2:
            raise ValueError(
                f"{self.reservoir_id}: "
                "at least two level-storage "
                "anchors are required"
            )

        if not np.all(
            np.isfinite(
                levels
            )
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "level anchors contain "
                "non-finite values"
            )

        if not np.all(
            np.isfinite(
                storages
            )
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "storage anchors contain "
                "non-finite values"
            )

        if np.any(
            storages < 0.0
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "storage anchors cannot "
                "be negative"
            )

        if not np.all(
            np.diff(
                levels
            )
            > 0.0
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "level anchors must be "
                "strictly increasing"
            )

        if not np.all(
            np.diff(
                storages
            )
            > 0.0
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                "storage anchors must be "
                "strictly increasing"
            )

        levels = levels.copy()
        storages = storages.copy()

        levels.setflags(
            write=False
        )

        storages.setflags(
            write=False
        )

        return (
            levels,
            storages,
        )

    def _strict_interp(
        self,
        values,
        x,
        y,
        value_name,
    ):
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                f"{value_name} contains "
                "non-finite values"
            )

        lower = float(
            x[
                0
            ]
        )

        upper = float(
            x[
                -1
            ]
        )

        tolerance = max(
            1e-10,
            abs(
                upper - lower
            )
            * 1e-12,
        )

        if np.any(
            array
            < (
                lower
                - tolerance
            )
        ):
            minimum = float(
                np.min(
                    array
                )
            )

            raise ValueError(
                f"{self.reservoir_id}: "
                f"{value_name}={minimum} "
                f"is below interpolation "
                f"range [{lower}, {upper}]"
            )

        if np.any(
            array
            > (
                upper
                + tolerance
            )
        ):
            maximum = float(
                np.max(
                    array
                )
            )

            raise ValueError(
                f"{self.reservoir_id}: "
                f"{value_name}={maximum} "
                f"is above interpolation "
                f"range [{lower}, {upper}]"
            )

        # Clip only values differing from an endpoint by
        # floating-point roundoff.  Substantive extrapolation
        # has already raised above.
        safe_array = np.clip(
            array,
            lower,
            upper,
        )

        result = np.interp(
            safe_array,
            x,
            y,
        )

        if array.ndim == 0:
            return float(
                result
            )

        return np.asarray(
            result,
            dtype=np.float64,
        )

    @staticmethod
    def _check_scalar_in_range(
        value,
        lower,
        upper,
        name,
    ):
        tolerance = max(
            1e-6,
            abs(
                upper - lower
            )
            * 1e-12,
        )

        if (
            value
            < lower - tolerance
            or value
            > upper + tolerance
        ):
            raise ValueError(
                f"{name}={value} outside "
                f"[{lower}, {upper}]"
            )

    def _positive_number(
        self,
        value,
        name,
    ) -> float:
        value = float(
            value
        )

        if (
            not math.isfinite(
                value
            )
            or value <= 0.0
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                f"{name} must be "
                "positive and finite"
            )

        return value

    def _nonnegative_number(
        self,
        value,
        name,
    ) -> float:
        value = float(
            value
        )

        if (
            not math.isfinite(
                value
            )
            or value < 0.0
        ):
            raise ValueError(
                f"{self.reservoir_id}: "
                f"{name} must be "
                "nonnegative and finite"
            )

        return value