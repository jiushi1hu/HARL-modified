from __future__ import annotations

from numbers import Integral

import numpy as np


class ReleaseActionMapper:

    def __init__(
        self,
        reservoir_id: str,
        num_actions: int,
        map_min_release_m3s: float,
        map_max_release_m3s: float,
    ):
        self.reservoir_id = str(reservoir_id)

        if not self.reservoir_id:
            raise ValueError(
                "reservoir_id cannot be empty"
            )

        if (
            not isinstance(num_actions, Integral)
            or isinstance(num_actions, (bool, np.bool_))
        ):
            raise TypeError(
                "num_actions must be an integer"
            )

        self.num_actions = int(num_actions)

        if self.num_actions < 2:
            raise ValueError(
                "num_actions must be at least 2"
            )

        self.map_min_release_m3s = (
            self._validate_nonnegative(
                map_min_release_m3s,
                "map_min_release_m3s",
            )
        )

        self.map_max_release_m3s = (
            self._validate_nonnegative(
                map_max_release_m3s,
                "map_max_release_m3s",
            )
        )

        if (
            self.map_min_release_m3s
            >= self.map_max_release_m3s
        ):
            raise ValueError(
                "map_min_release_m3s must be "
                "smaller than map_max_release_m3s"
            )

        self._normalized_actions = np.linspace(
            0.0,
            1.0,
            self.num_actions,
            dtype=np.float64,
        )

        self._candidate_releases_m3s = (
            self._map_normalized_actions(
                self._normalized_actions
            )
        )

        self._candidate_releases_m3s[0] = (
            self.map_min_release_m3s
        )
        self._candidate_releases_m3s[-1] = (
            self.map_max_release_m3s
        )

        if not np.all(
            np.diff(
                self._candidate_releases_m3s
            ) > 0.0
        ):
            raise RuntimeError(
                "release mapping must be "
                "strictly increasing"
            )

        self._normalized_actions.setflags(
            write=False
        )
        self._candidate_releases_m3s.setflags(
            write=False
        )

    @property
    def normalized_actions(
        self,
    ) -> np.ndarray:
        return self._normalized_actions.copy()

    @property
    def candidate_releases_m3s(
        self,
    ) -> np.ndarray:
        return (
            self._candidate_releases_m3s.copy()
        )

    def normalize_action(
        self,
        action_index: int,
    ) -> float:
        action_index = (
            self._validate_action_index(
                action_index
            )
        )

        return float(
            self._normalized_actions[
                action_index
            ]
        )

    def action_to_release(
        self,
        action_index: int,
    ) -> float:
        action_index = (
            self._validate_action_index(
                action_index
            )
        )

        return float(
            self._candidate_releases_m3s[
                action_index
            ]
        )

    def _map_normalized_actions(
        self,
        normalized_actions: np.ndarray,
    ) -> np.ndarray:
        return (
            self.map_min_release_m3s
            + (
                self.map_max_release_m3s
                - self.map_min_release_m3s
            )
            * normalized_actions
        )

    def _validate_action_index(
        self,
        action_index: int,
    ) -> int:
        if (
            not isinstance(action_index, Integral)
            or isinstance(
                action_index,
                (bool, np.bool_),
            )
        ):
            raise TypeError(
                "action_index must be an integer"
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
                f"action_index must be in "
                f"[0, {self.num_actions - 1}]"
            )

        return action_index

    @staticmethod
    def _validate_nonnegative(
        value: float,
        name: str,
    ) -> float:
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
