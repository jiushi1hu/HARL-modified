from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from harl.envs.cascade_reservoir.reservoir_physics import (
    ReservoirPhysics,
)


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_EXPECTED_ORDER = np.asarray(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    dtype=np.float64,
)

_INFLOW_COLUMNS = (
    "WDD_external_inflow_m3s",
    "BHT_interval_inflow_m3s",
    "XLD_interval_inflow_m3s",
    "XJB_interval_inflow_m3s",
    "THR_interval_inflow_m3s",
)

_LEVEL_LIMIT_COLUMNS = (
    "hard_min_level_m",
    "normal_upper_level_m",
    "flood_limit_level_m",
    "safety_upper_level_m",
)


class CascadeDataLoader:
    """Load and validate all data for the five-reservoir cascade."""

    def __init__(
        self,
        env_args: Mapping,
    ):
        if not isinstance(
            env_args,
            Mapping,
        ):
            raise TypeError(
                "env_args must be a mapping"
            )

        self.env_args = dict(
            env_args
        )

        self.reservoir_order = tuple(
            self.env_args[
                "reservoir_order"
            ]
        )

        if (
            self.reservoir_order
            != _RESERVOIR_ORDER
        ):
            raise ValueError(
                "reservoir_order must be "
                "WDD, BHT, XLD, XJB, THR"
            )

        self._load_scalar_config()
        self._resolve_paths()

        self.reservoir_physics_table = (
            self._load_reservoir_physics_table()
        )

        self.level_storage_anchors = (
            self._load_level_storage_anchors()
        )

        self.level_limits = (
            self._load_level_limits()
        )

        self.initial_storage_m3 = (
            self._load_initial_state()
        )

        self.daily_inflow = (
            self._load_daily_inflow()
        )

        self.physics = (
            self._build_physics()
        )

        self._cross_validate()

    @property
    def num_days(
        self,
    ) -> int:
        return len(
            self.daily_inflow
        )

    def _load_scalar_config(
        self,
    ) -> None:
        simulation = self.env_args[
            "simulation"
        ]

        hydropower = self.env_args[
            "hydropower"
        ]

        self.start_date = pd.Timestamp(
            simulation[
                "start_date"
            ]
        ).normalize()

        self.end_date = pd.Timestamp(
            simulation[
                "end_date"
            ]
        ).normalize()

        if (
            pd.isna(
                self.start_date
            )
            or pd.isna(
                self.end_date
            )
        ):
            raise ValueError(
                "simulation dates are invalid"
            )

        if (
            self.end_date
            < self.start_date
        ):
            raise ValueError(
                "simulation.end_date must not "
                "precede simulation.start_date"
            )

        self.timestep_seconds = float(
            simulation[
                "timestep_seconds"
            ]
        )

        if (
            not math.isfinite(
                self.timestep_seconds
            )
            or self.timestep_seconds <= 0.0
        ):
            raise ValueError(
                "simulation.timestep_seconds "
                "must be positive and finite"
            )

        self.efficiency = float(
            hydropower[
                "efficiency"
            ]
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
                "hydropower.efficiency "
                "must lie in (0, 1]"
            )

    def _resolve_paths(
        self,
    ) -> None:
        data_config = self.env_args[
            "data"
        ]

        if not isinstance(
            data_config,
            Mapping,
        ):
            raise TypeError(
                "env_args['data'] must "
                "be a mapping"
            )

        project_root = (
            Path(
                __file__
            )
            .resolve()
            .parents[
                3
            ]
        )

        root_config = Path(
            str(
                data_config[
                    "root"
                ]
            )
        ).expanduser()

        if root_config.is_absolute():
            data_root = (
                root_config.resolve()
            )
        else:
            data_root = (
                project_root
                / root_config
            ).resolve()

        if not data_root.exists():
            raise FileNotFoundError(
                "cascade reservoir data root "
                f"does not exist: {data_root}"
            )

        if not data_root.is_dir():
            raise NotADirectoryError(
                "cascade reservoir data root "
                f"is not a directory: {data_root}"
            )

        self.data_root = (
            data_root
        )

        required_files = (
            "reservoir_physics",
            "level_storage_anchors",
            "daily_inflow",
            "initial_state",
            "water_level_limits",
        )

        self.paths = {}

        for key in required_files:
            if key not in data_config:
                raise KeyError(
                    "data config missing "
                    f"`{key}`"
                )

            configured_path = Path(
                str(
                    data_config[
                        key
                    ]
                )
            ).expanduser()

            if configured_path.is_absolute():
                path = (
                    configured_path.resolve()
                )
            else:
                path = (
                    self.data_root
                    / configured_path
                ).resolve()

            if not path.exists():
                raise FileNotFoundError(
                    f"{key} file does not "
                    f"exist: {path}"
                )

            if not path.is_file():
                raise FileNotFoundError(
                    f"{key} is not a file: "
                    f"{path}"
                )

            self.paths[
                key
            ] = path

    def _load_reservoir_physics_table(
        self,
    ) -> pd.DataFrame:
        frame = self._read_csv(
            self.paths[
                "reservoir_physics"
            ],
            "reservoir_physics",
        )

        required = (
            "reservoir_id",
            "order",
            "installed_capacity_mw",
            "max_turbine_flow_m3s",
            "max_total_release_m3s",
            "typical_net_head_m",
        )

        self._require_columns(
            frame,
            required,
            "reservoir_physics.csv",
        )

        frame = frame.copy()

        frame[
            "reservoir_id"
        ] = self._clean_reservoir_ids(
            frame[
                "reservoir_id"
            ],
            "reservoir_physics.csv",
        )

        self._validate_exact_reservoir_set(
            frame[
                "reservoir_id"
            ],
            "reservoir_physics.csv",
        )

        order_values = pd.to_numeric(
            frame[
                "order"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                order_values
            )
        ):
            raise ValueError(
                "reservoir_physics.csv: "
                "order contains non-finite values"
            )

        # Important:
        # never use astype(int) here because
        # 1.8 would otherwise silently become 1.
        if not np.all(
            order_values
            == np.round(
                order_values
            )
        ):
            raise ValueError(
                "reservoir_physics.csv: "
                "order values must be exact integers"
            )

        sorted_indices = np.argsort(
            order_values
        )

        sorted_orders = (
            order_values[
                sorted_indices
            ]
        )

        if not np.array_equal(
            sorted_orders,
            _EXPECTED_ORDER,
        ):
            raise ValueError(
                "reservoir_physics.csv: "
                "order must be exactly "
                "1, 2, 3, 4, 5"
            )

        sorted_ids = tuple(
            frame.iloc[
                sorted_indices
            ][
                "reservoir_id"
            ].tolist()
        )

        if (
            sorted_ids
            != _RESERVOIR_ORDER
        ):
            raise ValueError(
                "reservoir_physics.csv: "
                "hydraulic order must be "
                "1=WDD, 2=BHT, 3=XLD, "
                "4=XJB, 5=THR"
            )

        numeric_columns = (
            "installed_capacity_mw",
            "max_turbine_flow_m3s",
            "max_total_release_m3s",
            "typical_net_head_m",
        )

        for column in numeric_columns:
            frame[
                column
            ] = pd.to_numeric(
                frame[
                    column
                ],
                errors="raise",
            )

            values = frame[
                column
            ].to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    values
                )
            ):
                raise ValueError(
                    "reservoir_physics.csv: "
                    f"{column} contains "
                    "non-finite values"
                )

            if np.any(
                values <= 0.0
            ):
                raise ValueError(
                    "reservoir_physics.csv: "
                    f"{column} must be positive"
                )

        if np.any(
            frame[
                "max_turbine_flow_m3s"
            ].to_numpy(
                dtype=np.float64
            )
            >
            frame[
                "max_total_release_m3s"
            ].to_numpy(
                dtype=np.float64
            )
        ):
            raise ValueError(
                "reservoir_physics.csv: "
                "max turbine flow cannot exceed "
                "max total release"
            )

        frame[
            "order"
        ] = order_values.astype(
            np.int64
        )

        frame = (
            frame.sort_values(
                "order"
            )
            .reset_index(
                drop=True
            )
        )

        return frame

    def _load_level_storage_anchors(
        self,
    ) -> pd.DataFrame:
        frame = self._read_csv(
            self.paths[
                "level_storage_anchors"
            ],
            "level_storage_anchors",
        )

        self._require_columns(
            frame,
            (
                "reservoir_id",
                "storage_m3",
            ),
            "level_storage_anchors.csv",
        )

        level_column = (
            self._resolve_single_column(
                frame=frame,
                candidates=(
                    "level_m",
                    "water_level_m",
                ),
                logical_name=(
                    "water level"
                ),
                file_name=(
                    "level_storage_anchors.csv"
                ),
            )
        )

        frame = frame.copy()

        if level_column != "level_m":
            frame = frame.rename(
                columns={
                    level_column:
                    "level_m"
                }
            )

        frame[
            "reservoir_id"
        ] = self._clean_reservoir_ids(
            frame[
                "reservoir_id"
            ],
            "level_storage_anchors.csv",
        )

        self._validate_exact_reservoir_set(
            frame[
                "reservoir_id"
            ],
            "level_storage_anchors.csv",
            require_one_row_each=False,
        )

        for column in (
            "level_m",
            "storage_m3",
        ):
            frame[
                column
            ] = pd.to_numeric(
                frame[
                    column
                ],
                errors="raise",
            )

            values = frame[
                column
            ].to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    values
                )
            ):
                raise ValueError(
                    "level_storage_anchors.csv: "
                    f"{column} contains "
                    "non-finite values"
                )

        if np.any(
            frame[
                "storage_m3"
            ].to_numpy(
                dtype=np.float64
            )
            < 0.0
        ):
            raise ValueError(
                "level_storage_anchors.csv: "
                "storage_m3 cannot be negative"
            )

        pieces = []

        for reservoir_id in (
            self.reservoir_order
        ):
            subset = (
                frame[
                    frame[
                        "reservoir_id"
                    ]
                    == reservoir_id
                ][
                    [
                        "reservoir_id",
                        "level_m",
                        "storage_m3",
                    ]
                ]
                .sort_values(
                    "level_m"
                )
                .reset_index(
                    drop=True
                )
            )

            if len(
                subset
            ) < 2:
                raise ValueError(
                    "level_storage_anchors.csv: "
                    f"{reservoir_id} requires "
                    "at least two anchors"
                )

            levels = subset[
                "level_m"
            ].to_numpy(
                dtype=np.float64
            )

            storages = subset[
                "storage_m3"
            ].to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.diff(
                    levels
                )
                > 0.0
            ):
                raise ValueError(
                    "level_storage_anchors.csv: "
                    f"{reservoir_id} levels must "
                    "be strictly increasing"
                )

            if not np.all(
                np.diff(
                    storages
                )
                > 0.0
            ):
                raise ValueError(
                    "level_storage_anchors.csv: "
                    f"{reservoir_id} storages must "
                    "be strictly increasing"
                )

            pieces.append(
                subset
            )

        return pd.concat(
            pieces,
            axis=0,
            ignore_index=True,
        )

    def _load_level_limits(
        self,
    ) -> pd.DataFrame:
        frame = self._read_csv(
            self.paths[
                "water_level_limits"
            ],
            "water_level_limits",
        )

        self._require_columns(
            frame,
            (
                "reservoir_id",
                *_LEVEL_LIMIT_COLUMNS,
            ),
            "water_level_limits.csv",
        )

        frame = frame.copy()

        frame[
            "reservoir_id"
        ] = self._clean_reservoir_ids(
            frame[
                "reservoir_id"
            ],
            "water_level_limits.csv",
        )

        self._validate_exact_reservoir_set(
            frame[
                "reservoir_id"
            ],
            "water_level_limits.csv",
        )

        for column in (
            _LEVEL_LIMIT_COLUMNS
        ):
            frame[
                column
            ] = pd.to_numeric(
                frame[
                    column
                ],
                errors="raise",
            )

            values = frame[
                column
            ].to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    values
                )
            ):
                raise ValueError(
                    "water_level_limits.csv: "
                    f"{column} contains "
                    "non-finite values"
                )

        frame = (
            self._sort_by_reservoir_order(
                frame
            )
        )

        for _, row in (
            frame.iterrows()
        ):
            reservoir_id = str(
                row[
                    "reservoir_id"
                ]
            )

            hard_min = float(
                row[
                    "hard_min_level_m"
                ]
            )

            flood_limit = float(
                row[
                    "flood_limit_level_m"
                ]
            )

            normal_upper = float(
                row[
                    "normal_upper_level_m"
                ]
            )

            safety_upper = float(
                row[
                    "safety_upper_level_m"
                ]
            )

            if not (
                hard_min
                <= flood_limit
                <= normal_upper
                <= safety_upper
            ):
                raise ValueError(
                    "water_level_limits.csv: "
                    f"{reservoir_id} must satisfy "
                    "hard_min <= flood_limit <= "
                    "normal_upper <= safety_upper"
                )

        return frame.reset_index(
            drop=True
        )

    def _load_initial_state(
        self,
    ):
        frame = self._read_csv(
            self.paths[
                "initial_state"
            ],
            "initial_state",
        )

        self._require_columns(
            frame,
            (
                "reservoir_id",
            ),
            "initial_state.csv",
        )

        storage_column = (
            self._resolve_single_column(
                frame=frame,
                candidates=(
                    "initial_storage_m3",
                    "storage_m3",
                ),
                logical_name=(
                    "initial storage"
                ),
                file_name=(
                    "initial_state.csv"
                ),
            )
        )

        frame = frame.copy()

        frame[
            "reservoir_id"
        ] = self._clean_reservoir_ids(
            frame[
                "reservoir_id"
            ],
            "initial_state.csv",
        )

        self._validate_exact_reservoir_set(
            frame[
                "reservoir_id"
            ],
            "initial_state.csv",
        )

        frame[
            storage_column
        ] = pd.to_numeric(
            frame[
                storage_column
            ],
            errors="raise",
        )

        values = frame[
            storage_column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                "initial_state.csv: "
                "initial storage contains "
                "non-finite values"
            )

        if np.any(
            values <= 0.0
        ):
            raise ValueError(
                "initial_state.csv: "
                "initial storage must "
                "be positive"
            )

        frame = (
            self._sort_by_reservoir_order(
                frame
            )
        )

        result = {}

        for _, row in (
            frame.iterrows()
        ):
            reservoir_id = str(
                row[
                    "reservoir_id"
                ]
            )

            result[
                reservoir_id
            ] = float(
                row[
                    storage_column
                ]
            )

        return result

    def _load_daily_inflow(
        self,
    ) -> pd.DataFrame:
        frame = self._read_csv(
            self.paths[
                "daily_inflow"
            ],
            "daily_inflow",
        )

        self._require_columns(
            frame,
            (
                "date",
                *_INFLOW_COLUMNS,
            ),
            "daily_inflow.csv",
        )

        frame = frame.copy()

        frame[
            "date"
        ] = pd.to_datetime(
            frame[
                "date"
            ],
            format="%Y-%m-%d",
            errors="raise",
        ).dt.normalize()

        if frame[
            "date"
        ].duplicated().any():
            duplicated = (
                frame.loc[
                    frame[
                        "date"
                    ].duplicated(
                        keep=False
                    ),
                    "date",
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .tolist()
            )

            raise ValueError(
                "daily_inflow.csv contains "
                "duplicate dates: "
                f"{duplicated[:5]}"
            )

        for column in (
            _INFLOW_COLUMNS
        ):
            frame[
                column
            ] = pd.to_numeric(
                frame[
                    column
                ],
                errors="raise",
            )

            values = frame[
                column
            ].to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    values
                )
            ):
                raise ValueError(
                    "daily_inflow.csv: "
                    f"{column} contains "
                    "non-finite values"
                )

            if np.any(
                values < 0.0
            ):
                raise ValueError(
                    "daily_inflow.csv: "
                    f"{column} cannot "
                    "be negative"
                )

        frame = (
            frame.sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        expected_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq="D",
        )

        actual_dates = pd.DatetimeIndex(
            frame[
                "date"
            ]
        )

        if len(
            actual_dates
        ) != len(
            expected_dates
        ):
            raise ValueError(
                "daily_inflow.csv must contain "
                "exactly one row for every day "
                f"from {self.start_date.date()} "
                f"through {self.end_date.date()}; "
                f"expected {len(expected_dates)} "
                f"rows, received "
                f"{len(actual_dates)}"
            )

        if not actual_dates.equals(
            expected_dates
        ):
            missing = (
                expected_dates.difference(
                    actual_dates
                )
            )

            extra = (
                actual_dates.difference(
                    expected_dates
                )
            )

            raise ValueError(
                "daily_inflow.csv date coverage "
                "does not exactly match the "
                "configured simulation period. "
                f"Missing={self._format_dates(missing)}, "
                f"extra={self._format_dates(extra)}"
            )

        return frame[
            [
                "date",
                *_INFLOW_COLUMNS,
            ]
        ].copy()

    def _build_physics(
        self,
    ):
        physics_table = (
            self.reservoir_physics_table
            .set_index(
                "reservoir_id",
                drop=False,
            )
        )

        result = {}

        for reservoir_id in (
            self.reservoir_order
        ):
            row = physics_table.loc[
                reservoir_id
            ]

            anchors = (
                self.level_storage_anchors[
                    self.level_storage_anchors[
                        "reservoir_id"
                    ]
                    == reservoir_id
                ]
                .sort_values(
                    "level_m"
                )
            )

            result[
                reservoir_id
            ] = ReservoirPhysics(
                reservoir_id=(
                    reservoir_id
                ),
                installed_capacity_mw=float(
                    row[
                        "installed_capacity_mw"
                    ]
                ),
                max_turbine_flow_m3s=float(
                    row[
                        "max_turbine_flow_m3s"
                    ]
                ),
                max_total_release_m3s=float(
                    row[
                        "max_total_release_m3s"
                    ]
                ),
                typical_net_head_m=float(
                    row[
                        "typical_net_head_m"
                    ]
                ),
                level_anchors_m=(
                    anchors[
                        "level_m"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
                storage_anchors_m3=(
                    anchors[
                        "storage_m3"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
                efficiency=(
                    self.efficiency
                ),
                timestep_seconds=(
                    self.timestep_seconds
                ),
            )

        return result

    def _cross_validate(
        self,
    ) -> None:
        limits = (
            self.level_limits
            .set_index(
                "reservoir_id"
            )
        )

        for reservoir_id in (
            self.reservoir_order
        ):
            physics = self.physics[
                reservoir_id
            ]

            row = limits.loc[
                reservoir_id
            ]

            for column in (
                _LEVEL_LIMIT_COLUMNS
            ):
                level = float(
                    row[
                        column
                    ]
                )

                if (
                    level
                    < physics.min_level_m
                    or level
                    > physics.max_level_m
                ):
                    raise ValueError(
                        "water_level_limits.csv: "
                        f"{reservoir_id} "
                        f"{column}={level} lies "
                        "outside the level-storage "
                        "curve range "
                        f"[{physics.min_level_m}, "
                        f"{physics.max_level_m}]"
                    )

                # Also explicitly verify that
                # interpolation is available.
                physics.storage_from_level(
                    level
                )

            initial_storage = float(
                self.initial_storage_m3[
                    reservoir_id
                ]
            )

            if (
                initial_storage
                < physics.min_storage_m3
                or initial_storage
                > physics.max_storage_m3
            ):
                raise ValueError(
                    "initial_state.csv: "
                    f"{reservoir_id} initial "
                    "storage lies outside the "
                    "level-storage curve"
                )

            initial_level = float(
                physics.level_from_storage(
                    initial_storage
                )
            )

            hard_min = float(
                row[
                    "hard_min_level_m"
                ]
            )

            safety_upper = float(
                row[
                    "safety_upper_level_m"
                ]
            )

            if not (
                hard_min
                <= initial_level
                <= safety_upper
            ):
                raise ValueError(
                    "initial_state.csv: "
                    f"{reservoir_id} initial "
                    f"level {initial_level} m "
                    "violates P1 interval "
                    f"[{hard_min}, "
                    f"{safety_upper}]"
                )

    @staticmethod
    def _read_csv(
        path: Path,
        logical_name: str,
    ) -> pd.DataFrame:
        try:
            frame = pd.read_csv(
                path,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            frame = pd.read_csv(
                path,
                encoding="utf-8-sig",
            )

        if frame.empty:
            raise ValueError(
                f"{logical_name} is empty: "
                f"{path}"
            )

        frame.columns = [
            str(
                column
            ).strip()
            for column
            in frame.columns
        ]

        return frame

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        columns: Sequence[str],
        file_name: str,
    ) -> None:
        missing = [
            column
            for column in columns
            if column
            not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{file_name} missing "
                f"required columns: "
                f"{missing}"
            )

    @staticmethod
    def _resolve_single_column(
        frame: pd.DataFrame,
        candidates: Sequence[str],
        logical_name: str,
        file_name: str,
    ) -> str:
        matches = [
            column
            for column
            in candidates
            if column
            in frame.columns
        ]

        if len(
            matches
        ) == 0:
            raise ValueError(
                f"{file_name} missing "
                f"{logical_name} column; "
                f"expected one of "
                f"{tuple(candidates)}"
            )

        if len(
            matches
        ) > 1:
            raise ValueError(
                f"{file_name} contains "
                f"multiple columns for "
                f"{logical_name}: "
                f"{matches}"
            )

        return matches[
            0
        ]

    @staticmethod
    def _clean_reservoir_ids(
        series: pd.Series,
        file_name: str,
    ) -> pd.Series:
        if series.isna().any():
            raise ValueError(
                f"{file_name}: reservoir_id "
                "contains missing values"
            )

        cleaned = series.astype(
            str
        ).str.strip()

        if (
            cleaned == ""
        ).any():
            raise ValueError(
                f"{file_name}: reservoir_id "
                "contains empty values"
            )

        return cleaned

    @staticmethod
    def _validate_exact_reservoir_set(
        series: pd.Series,
        file_name: str,
        require_one_row_each: bool = True,
    ) -> None:
        values = series.tolist()

        unique_values = set(
            values
        )

        expected = set(
            _RESERVOIR_ORDER
        )

        if (
            unique_values
            != expected
        ):
            missing = sorted(
                expected
                - unique_values
            )

            extra = sorted(
                unique_values
                - expected
            )

            raise ValueError(
                f"{file_name}: reservoir set "
                "must be exactly "
                "WDD, BHT, XLD, XJB, THR. "
                f"Missing={missing}, "
                f"extra={extra}"
            )

        if require_one_row_each:
            counts = series.value_counts()

            invalid = {
                reservoir_id:
                int(
                    counts.get(
                        reservoir_id,
                        0,
                    )
                )
                for reservoir_id
                in _RESERVOIR_ORDER
                if int(
                    counts.get(
                        reservoir_id,
                        0,
                    )
                )
                != 1
            }

            if invalid:
                raise ValueError(
                    f"{file_name}: expected "
                    "exactly one row per "
                    f"reservoir, got {invalid}"
                )

    @staticmethod
    def _sort_by_reservoir_order(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        position = {
            reservoir_id:
            index
            for index, reservoir_id
            in enumerate(
                _RESERVOIR_ORDER
            )
        }

        result = frame.copy()

        result[
            "_reservoir_position"
        ] = result[
            "reservoir_id"
        ].map(
            position
        )

        result = (
            result.sort_values(
                "_reservoir_position"
            )
            .drop(
                columns=[
                    "_reservoir_position"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    @staticmethod
    def _format_dates(
        dates,
    ):
        if len(
            dates
        ) == 0:
            return []

        return [
            pd.Timestamp(
                value
            ).strftime(
                "%Y-%m-%d"
            )
            for value
            in dates[
                :5
            ]
        ]