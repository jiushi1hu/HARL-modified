from __future__ import annotations

import math
from datetime import date
from numbers import Integral
from typing import Mapping, Sequence

import gym
import numpy as np
import pandas as pd

from harl.envs.cascade_reservoir.action_mapping import (
    ReleaseActionMapper,
)
from harl.envs.cascade_reservoir.constraint_costs import (
    ConstraintCostEvaluator,
)
from harl.envs.cascade_reservoir.data_loader import (
    CascadeDataLoader,
)
from harl.envs.cascade_reservoir.routing import (
    compute_downstream_inflow,
)
from harl.envs.cascade_reservoir.safety.constraint_context import (
    LevelBounds,
    P1Constraints,
    P2Constraints,
    ReleaseBounds,
)
from harl.envs.cascade_reservoir.s3_relaxation import (
    ReservoirSafetyState,
    S3RelaxationSolver,
    SafetyRecoveryResult,
)


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_DOWNSTREAM_ORDER = (
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_INFLOW_COLUMNS = {
    "WDD": "WDD_external_inflow_m3s",
    "BHT": "BHT_interval_inflow_m3s",
    "XLD": "XLD_interval_inflow_m3s",
    "XJB": "XJB_interval_inflow_m3s",
    "THR": "THR_interval_inflow_m3s",
}

_BASE_OBS_DIM = 17
_DOWNSTREAM_DECISION_OBS_DIM = 18
_SHARE_OBS_DIM = (
    len(_RESERVOIR_ORDER)
    * _BASE_OBS_DIM
)

_SECONDS_PER_HOUR = 3600.0

_FLOW_TOLERANCE_M3S = 1e-6
_LEVEL_TOLERANCE_M = 1e-5
_STORAGE_TOLERANCE_M3 = 1.0


class CascadeReservoirEnv:
    """Five-reservoir cascade scheduling environment.

    Hydraulic order:

        WDD -> BHT -> XLD -> XJB -> THR

    prepare_step() constructs the current S1-S3 decision
    context.

    Actual policy sampling is performed upstream-to-downstream
    by SequentialSampler.

    step() executes the already selected joint action and
    advances the physical reservoir states.
    """

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
            str(
                reservoir_id
            )
            for reservoir_id
            in self.env_args[
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

        self.n_agents = len(
            self.reservoir_order
        )

        if (
            self.env_args.get(
                "state_type",
                "EP",
            )
            != "EP"
        ):
            raise ValueError(
                "cascade_reservoir requires "
                "state_type = EP"
            )

        self._load_simulation_config()
        self._load_data()
        self._build_physics_metadata()
        self._build_action_mappers()
        self._build_constraints()
        self._build_safety_solver()
        self._build_long_term_constraints()
        self._build_spaces()

        self._rng = (
            np.random.default_rng()
        )

        self._date_index = 0
        self._terminated = False

        self.storage_m3 = {}
        self.previous_action = {}
        self.previous_release_m3s = {}
        self.previous_power_mw = {}

        self._prepared_context = None

        self.reset()

    def _load_simulation_config(
        self,
    ) -> None:
        simulation = self.env_args[
            "simulation"
        ]

        self.start_date = (
            date.fromisoformat(
                str(
                    simulation[
                        "start_date"
                    ]
                )
            )
        )

        self.end_date = (
            date.fromisoformat(
                str(
                    simulation[
                        "end_date"
                    ]
                )
            )
        )

        if (
            self.end_date
            < self.start_date
        ):
            raise ValueError(
                "simulation.end_date "
                "must not precede start_date"
            )

        self.timestep_seconds = (
            _positive_finite(
                simulation[
                    "timestep_seconds"
                ],
                (
                    "simulation."
                    "timestep_seconds"
                ),
            )
        )

        self.timestep_hours = (
            self.timestep_seconds
            / _SECONDS_PER_HOUR
        )

        self.operation_stage_config = (
            dict(
                self.env_args[
                    "operation_stage"
                ]
            )
        )

        self.level_control_config = (
            dict(
                self.env_args[
                    "level_control"
                ]
            )
        )

        if not self.operation_stage_config:
            raise ValueError(
                "operation_stage "
                "cannot be empty"
            )

        if not self.level_control_config:
            raise ValueError(
                "level_control "
                "cannot be empty"
            )

    def _load_data(
        self,
    ) -> None:
        self.data_loader = (
            CascadeDataLoader(
                self.env_args
            )
        )

        required_attributes = (
            "physics",
            "daily_inflow",
            "initial_storage_m3",
            "level_limits",
        )

        for attribute in (
            required_attributes
        ):
            if not hasattr(
                self.data_loader,
                attribute,
            ):
                raise AttributeError(
                    "CascadeDataLoader "
                    "must expose "
                    f"`{attribute}`"
                )

        self.physics = dict(
            self.data_loader.physics
        )

        if (
            set(
                self.physics
            )
            != set(
                self.reservoir_order
            )
        ):
            raise ValueError(
                "physics data must contain "
                "exactly WDD, BHT, XLD, "
                "XJB and THR"
            )

        self.physics = {
            reservoir_id:
            self.physics[
                reservoir_id
            ]
            for reservoir_id
            in self.reservoir_order
        }

        self.daily_inflow = (
            self._prepare_daily_inflow(
                self.data_loader
                .daily_inflow
            )
        )

        self.initial_storage_m3 = {
            reservoir_id:
            self._get_initial_storage(
                reservoir_id
            )
            for reservoir_id
            in self.reservoir_order
        }

        self.level_limits = {
            reservoir_id:
            self._get_level_limit_row(
                reservoir_id
            )
            for reservoir_id
            in self.reservoir_order
        }

    def _prepare_daily_inflow(
        self,
        inflow_data,
    ) -> pd.DataFrame:
        if not isinstance(
            inflow_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "daily_inflow must be "
                "a pandas DataFrame"
            )

        frame = (
            inflow_data.copy()
        )

        if "date" in frame.columns:
            frame[
                "date"
            ] = pd.to_datetime(
                frame[
                    "date"
                ],
                errors="raise",
            )

            frame = frame.set_index(
                "date"
            )

        elif not isinstance(
            frame.index,
            pd.DatetimeIndex,
        ):
            raise ValueError(
                "daily_inflow must "
                "contain a date column "
                "or DatetimeIndex"
            )

        frame.index = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    frame.index,
                    errors="raise",
                )
            )
            .normalize()
        )

        if (
            frame.index
            .duplicated()
            .any()
        ):
            raise ValueError(
                "daily_inflow contains "
                "duplicate dates"
            )

        required_columns = tuple(
            _INFLOW_COLUMNS.values()
        )

        missing_columns = [
            column
            for column
            in required_columns
            if column
            not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                "daily_inflow missing "
                f"columns: {missing_columns}"
            )

        expected_dates = (
            pd.date_range(
                self.start_date,
                self.end_date,
                freq="D",
            )
        )

        missing_dates = (
            expected_dates.difference(
                frame.index
            )
        )

        if len(
            missing_dates
        ) > 0:
            raise ValueError(
                "daily_inflow does not "
                "cover the complete "
                "simulation period"
            )

        frame = frame.loc[
            expected_dates,
            list(
                required_columns
            ),
        ].copy()

        values = frame.to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                "daily_inflow contains "
                "non-finite values"
            )

        if np.any(
            values < 0.0
        ):
            raise ValueError(
                "daily inflow cannot "
                "be negative"
            )

        return frame

    def _build_physics_metadata(
        self,
    ) -> None:
        self.hard_min_level_m = {}
        self.safety_upper_level_m = {}

        self.min_storage_m3 = {}
        self.max_storage_m3 = {}

        self.max_total_release_m3s = {}
        self.installed_capacity_mw = {}

        for reservoir_id in (
            self.reservoir_order
        ):
            physics = self.physics[
                reservoir_id
            ]

            physics_id = str(
                getattr(
                    physics,
                    "reservoir_id",
                    "",
                )
            )

            if (
                physics_id
                != reservoir_id
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "physics reservoir_id "
                    "mismatch: "
                    f"{physics_id!r}"
                )

            physics_timestep = (
                _positive_finite(
                    getattr(
                        physics,
                        "timestep_seconds",
                        None,
                    ),
                    (
                        f"{reservoir_id}."
                        "physics."
                        "timestep_seconds"
                    ),
                )
            )

            if not math.isclose(
                physics_timestep,
                self.timestep_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "physics timestep "
                    "does not match "
                    "environment timestep"
                )

            for method_name in (
                "storage_from_level",
                "level_from_storage",
                "transition",
            ):
                if not hasattr(
                    physics,
                    method_name,
                ):
                    raise AttributeError(
                        f"{reservoir_id}: "
                        "physics must "
                        "implement "
                        f"{method_name}()"
                    )

            row = self.level_limits[
                reservoir_id
            ]

            hard_min = _finite(
                row[
                    "hard_min_level_m"
                ],
                (
                    f"{reservoir_id}."
                    "hard_min_level_m"
                ),
            )

            safety_max = _finite(
                row[
                    "safety_upper_level_m"
                ],
                (
                    f"{reservoir_id}."
                    "safety_upper_level_m"
                ),
            )

            if (
                safety_max
                <= hard_min
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "invalid P1 "
                    "level limits"
                )

            min_storage = float(
                physics.storage_from_level(
                    hard_min
                )
            )

            max_storage = float(
                physics.storage_from_level(
                    safety_max
                )
            )

            if not (
                math.isfinite(
                    min_storage
                )
                and math.isfinite(
                    max_storage
                )
                and max_storage
                > min_storage
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "invalid P1 "
                    "storage interval"
                )

            max_release = (
                _positive_finite(
                    getattr(
                        physics,
                        "max_total_release_m3s",
                        None,
                    ),
                    (
                        f"{reservoir_id}."
                        "max_total_release_m3s"
                    ),
                )
            )

            installed_capacity = (
                _positive_finite(
                    getattr(
                        physics,
                        "installed_capacity_mw",
                        None,
                    ),
                    (
                        f"{reservoir_id}."
                        "installed_capacity_mw"
                    ),
                )
            )

            self.hard_min_level_m[
                reservoir_id
            ] = hard_min

            self.safety_upper_level_m[
                reservoir_id
            ] = safety_max

            self.min_storage_m3[
                reservoir_id
            ] = min_storage

            self.max_storage_m3[
                reservoir_id
            ] = max_storage

            self.max_total_release_m3s[
                reservoir_id
            ] = max_release

            self.installed_capacity_mw[
                reservoir_id
            ] = installed_capacity

        self.total_installed_capacity_mw = (
            float(
                sum(
                    self
                    .installed_capacity_mw
                    .values()
                )
            )
        )

        self.reward_reference_mwh = (
            self.total_installed_capacity_mw
            * self.timestep_hours
        )

        if (
            self.reward_reference_mwh
            <= 0.0
        ):
            raise ValueError(
                "reward reference "
                "must be positive"
            )

    def _build_action_mappers(
        self,
    ) -> None:
        num_actions = (
            self.env_args[
                "action"
            ][
                "num_actions"
            ]
        )

        if (
            not isinstance(
                num_actions,
                Integral,
            )
            or isinstance(
                num_actions,
                (bool, np.bool_),
            )
        ):
            raise TypeError(
                "action.num_actions "
                "must be an integer"
            )

        self.num_actions = int(
            num_actions
        )

        if (
            self.num_actions
            < 2
        ):
            raise ValueError(
                "action.num_actions "
                "must be at least 2"
            )

        self.action_mappers = {
            reservoir_id:
            ReleaseActionMapper(
                reservoir_id=(
                    reservoir_id
                ),
                num_actions=(
                    self.num_actions
                ),
                map_min_release_m3s=0.0,
                map_max_release_m3s=(
                    self
                    .max_total_release_m3s[
                        reservoir_id
                    ]
                ),
            )
            for reservoir_id
            in self.reservoir_order
        }

    def _build_constraints(
        self,
    ) -> None:
        self.p1_constraints = {
            reservoir_id:
            P1Constraints(
                level=LevelBounds(
                    min_level_m=(
                        self
                        .hard_min_level_m[
                            reservoir_id
                        ]
                    ),
                    max_level_m=(
                        self
                        .safety_upper_level_m[
                            reservoir_id
                        ]
                    ),
                ),
                release=ReleaseBounds(
                    min_release_m3s=0.0,
                    max_release_m3s=(
                        self
                        .max_total_release_m3s[
                            reservoir_id
                        ]
                    ),
                ),
            )
            for reservoir_id
            in self.reservoir_order
        }

    def _build_safety_solver(
        self,
    ) -> None:
        safety = self.env_args[
            "safety"
        ]

        self.s3_solver = (
            S3RelaxationSolver(
                reservoir_order=(
                    self.reservoir_order
                ),
                p3_config=dict(
                    safety[
                        "p3"
                    ]
                ),
                p4_config=dict(
                    safety[
                        "p4"
                    ]
                ),
                relaxation_config=dict(
                    safety[
                        "relaxation"
                    ]
                ),
            )
        )

    def _build_long_term_constraints(
        self,
    ) -> None:
        self.constraint_cost_evaluator = (
            ConstraintCostEvaluator
            .from_config(
                self.env_args
            )
        )

    def _build_spaces(
        self,
    ) -> None:
        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    _BASE_OBS_DIM,
                ),
                dtype=np.float32,
            )
        ]

        for _ in (
            _DOWNSTREAM_ORDER
        ):
            self.observation_space.append(
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        _DOWNSTREAM_DECISION_OBS_DIM,
                    ),
                    dtype=np.float32,
                )
            )

        share_space = (
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    _SHARE_OBS_DIM,
                ),
                dtype=np.float32,
            )
        )

        self.share_observation_space = [
            share_space
            for _ in range(
                self.n_agents
            )
        ]

        self.action_space = [
            gym.spaces.Discrete(
                self.action_mappers[
                    reservoir_id
                ].num_actions
            )
            for reservoir_id
            in self.reservoir_order
        ]

    def reset(
        self,
    ):
        self._date_index = 0
        self._terminated = False

        self.storage_m3 = {
            reservoir_id:
            float(
                self.initial_storage_m3[
                    reservoir_id
                ]
            )
            for reservoir_id
            in self.reservoir_order
        }

        for reservoir_id in (
            self.reservoir_order
        ):
            storage = (
                self.storage_m3[
                    reservoir_id
                ]
            )

            if not (
                self.min_storage_m3[
                    reservoir_id
                ]
                <= storage
                <= self.max_storage_m3[
                    reservoir_id
                ]
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "initial storage "
                    "violates P1"
                )

            self.physics[
                reservoir_id
            ].level_from_storage(
                storage
            )

        # No verified pre-simulation
        # action/release/power exists.
        self.previous_action = {
            reservoir_id: None
            for reservoir_id
            in self.reservoir_order
        }

        self.previous_release_m3s = {
            reservoir_id: None
            for reservoir_id
            in self.reservoir_order
        }

        self.previous_power_mw = {
            reservoir_id: None
            for reservoir_id
            in self.reservoir_order
        }

        self._prepared_context = None

        return (
            self
            ._build_interface_observations(),
            self
            ._build_share_observations(),
            self.get_avail_actions(),
        )

    def prepare_step(
        self,
    ):
        """Prepare a read-only S1-S3 context before S4 sampling."""

        if self._terminated:
            raise RuntimeError(
                "episode has terminated; "
                "call reset() before "
                "prepare_step()"
            )

        if (
            self._prepared_context
            is not None
        ):
            return (
                self._copy_prepared_context(
                    self._prepared_context
                )
            )

        current_date = (
            self.current_date
        )

        stage = (
            self._resolve_operation_stage(
                current_date
            )
        )

        base_observations = (
            self._build_base_observations()
        )

        forcing = (
            self._get_current_forcing()
        )

        safety_states = []

        for reservoir_id in (
            self.reservoir_order
        ):
            if (
                reservoir_id
                == "WDD"
            ):
                forcing_inflow = (
                    forcing[
                        "external_inflow_m3s"
                    ]
                )
            else:
                forcing_inflow = (
                    forcing[
                        "interval_inflows_m3s"
                    ][
                        reservoir_id
                    ]
                )

            safety_states.append(
                ReservoirSafetyState(
                    reservoir_id=(
                        reservoir_id
                    ),
                    physics=(
                        self.physics[
                            reservoir_id
                        ]
                    ),
                    mapper=(
                        self.action_mappers[
                            reservoir_id
                        ]
                    ),
                    storage_m3=(
                        self.storage_m3[
                            reservoir_id
                        ]
                    ),
                    forcing_inflow_m3s=(
                        forcing_inflow
                    ),
                    previous_release_m3s=(
                        self
                        .previous_release_m3s[
                            reservoir_id
                        ]
                    ),
                    p1=(
                        self.p1_constraints[
                            reservoir_id
                        ]
                    ),
                    p2=(
                        self
                        ._get_p2_constraints(
                            reservoir_id,
                            stage,
                        )
                    ),
                )
            )

        safety_result = (
            self.s3_solver.solve(
                states=tuple(
                    safety_states
                ),
                operation_stage=stage,
            )
        )

        if not isinstance(
            safety_result,
            SafetyRecoveryResult,
        ):
            raise TypeError(
                "S3RelaxationSolver.solve() "
                "must return "
                "SafetyRecoveryResult"
            )

        expected_num_actions = tuple(
            self.action_mappers[
                reservoir_id
            ].num_actions
            for reservoir_id
            in self.reservoir_order
        )

        if (
            tuple(
                safety_result
                .num_actions
            )
            != expected_num_actions
        ):
            raise RuntimeError(
                "safety result action "
                "dimensions do not match "
                "environment action mappers"
            )

        self._prepared_context = {
            "date": str(
                current_date
            ),
            "operation_stage": (
                stage
            ),
            "base_observations": {
                reservoir_id:
                base_observations[
                    reservoir_id
                ][
                    np.newaxis,
                    :
                ].copy()
                for reservoir_id
                in self.reservoir_order
            },
            "share_observation": (
                self._build_global_state(
                    base_observations
                )[
                    np.newaxis,
                    :
                ].copy()
            ),
            "external_inflow_m3s": float(
                forcing[
                    "external_inflow_m3s"
                ]
            ),
            "interval_inflows_m3s": {
                reservoir_id:
                float(
                    forcing[
                        "interval_inflows_m3s"
                    ][
                        reservoir_id
                    ]
                )
                for reservoir_id
                in _DOWNSTREAM_ORDER
            },
            "safety_result": (
                safety_result
            ),
        }

        return (
            self._copy_prepared_context(
                self._prepared_context
            )
        )

    def step(
        self,
        actions,
    ):
        if self._terminated:
            raise RuntimeError(
                "episode has terminated; "
                "call reset() before step()"
            )

        if (
            self._prepared_context
            is None
        ):
            self.prepare_step()

        context = (
            self._prepared_context
        )

        action_indices = (
            self._parse_actions(
                actions
            )
        )

        safety_result = (
            context[
                "safety_result"
            ]
        )

        # Do not allow direct env.step()
        # to bypass S2/S3.
        self._validate_safe_joint_action(
            action_indices,
            safety_result,
        )

        target_releases = {
            reservoir_id:
            float(
                self.action_mappers[
                    reservoir_id
                ].action_to_release(
                    action_indices[
                        reservoir_id
                    ]
                )
            )
            for reservoir_id
            in self.reservoir_order
        }

        actual_inflows = {}
        transitions = {}

        external_inflow = float(
            context[
                "external_inflow_m3s"
            ]
        )

        interval_inflows = (
            context[
                "interval_inflows_m3s"
            ]
        )

        upstream_release = None

        for reservoir_id in (
            self.reservoir_order
        ):
            if (
                reservoir_id
                == "WDD"
            ):
                current_inflow = (
                    external_inflow
                )

            else:
                current_inflow = float(
                    compute_downstream_inflow(
                        upstream_flow=(
                            upstream_release
                        ),
                        interval_inflow=(
                            interval_inflows[
                                reservoir_id
                            ]
                        ),
                    )
                )

            current_inflow = (
                _nonnegative_finite(
                    current_inflow,
                    (
                        f"{reservoir_id}."
                        "actual_inflow_m3s"
                    ),
                )
            )

            actual_inflows[
                reservoir_id
            ] = current_inflow

            release = (
                target_releases[
                    reservoir_id
                ]
            )

            transition = (
                self.physics[
                    reservoir_id
                ].transition(
                    storage_m3=(
                        self.storage_m3[
                            reservoir_id
                        ]
                    ),
                    inflow_m3s=(
                        current_inflow
                    ),
                    total_release_m3s=(
                        release
                    ),
                    timestep_seconds=(
                        self.timestep_seconds
                    ),
                )
            )

            self._validate_transition_p1(
                reservoir_id,
                release,
                transition,
            )

            transitions[
                reservoir_id
            ] = transition

            # Current implementation:
            #
            # Q_exec == Q_target
            upstream_release = release

        executed_releases = {
            reservoir_id:
            float(
                target_releases[
                    reservoir_id
                ]
            )
            for reservoir_id
            in self.reservoir_order
        }

        powers_mw = {
            reservoir_id:
            float(
                transitions[
                    reservoir_id
                ].power_mw
            )
            for reservoir_id
            in self.reservoir_order
        }

        energies_mwh = {
            reservoir_id:
            float(
                transitions[
                    reservoir_id
                ].energy_mwh
            )
            for reservoir_id
            in self.reservoir_order
        }

        system_energy_mwh = (
            float(
                sum(
                    energies_mwh.values()
                )
            )
        )

        reward = float(
            system_energy_mwh
            / self.reward_reference_mwh
        )

        if not math.isfinite(
            reward
        ):
            raise RuntimeError(
                "non-finite system reward"
            )

        constraint_result = (
            self
            .constraint_cost_evaluator
            .evaluate(
                executed_releases_m3s=(
                    executed_releases
                ),
                powers_mw=(
                    powers_mw
                ),
                operation_stage=(
                    context[
                        "operation_stage"
                    ]
                ),
            )
        )

        executed_date = (
            self.current_date
        )

        infos = (
            self._build_infos(
                executed_date=(
                    executed_date
                ),
                operation_stage=(
                    context[
                        "operation_stage"
                    ]
                ),
                action_indices=(
                    action_indices
                ),
                actual_inflows=(
                    actual_inflows
                ),
                interval_inflows=(
                    interval_inflows
                ),
                target_releases=(
                    target_releases
                ),
                executed_releases=(
                    executed_releases
                ),
                transitions=(
                    transitions
                ),
                system_energy_mwh=(
                    system_energy_mwh
                ),
                reward=reward,
                safety_result=(
                    safety_result
                ),
                constraint_result=(
                    constraint_result
                ),
            )
        )

        for reservoir_id in (
            self.reservoir_order
        ):
            transition = (
                transitions[
                    reservoir_id
                ]
            )

            self.storage_m3[
                reservoir_id
            ] = float(
                transition
                .next_storage_m3
            )

            self.previous_action[
                reservoir_id
            ] = int(
                action_indices[
                    reservoir_id
                ]
            )

            self.previous_release_m3s[
                reservoir_id
            ] = float(
                executed_releases[
                    reservoir_id
                ]
            )

            self.previous_power_mw[
                reservoir_id
            ] = float(
                powers_mw[
                    reservoir_id
                ]
            )

        done = bool(
            self._date_index
            == len(
                self.daily_inflow
            ) - 1
        )

        if done:
            self._terminated = True

        else:
            self._date_index += 1

        self._prepared_context = None

        rewards = [
            [
                reward
            ]
            for _ in range(
                self.n_agents
            )
        ]

        dones = [
            done
            for _ in range(
                self.n_agents
            )
        ]

        return (
            self
            ._build_interface_observations(),
            self
            ._build_share_observations(),
            rewards,
            dones,
            infos,
            self.get_avail_actions(),
        )

    def get_action_mappers(
        self,
    ):
        return {
            reservoir_id:
            self.action_mappers[
                reservoir_id
            ]
            for reservoir_id
            in self.reservoir_order
        }

    def get_avail_actions(
        self,
    ):
        # These are only HARL vector-env
        # interface placeholders.
        #
        # The real behavior-time masks are
        # generated by S2/S3 in prepare_step().
        return [
            np.ones(
                self.action_mappers[
                    reservoir_id
                ].num_actions,
                dtype=np.float32,
            )
            for reservoir_id
            in self.reservoir_order
        ]

    @property
    def current_date(
        self,
    ):
        return (
            self.daily_inflow
            .index[
                self._date_index
            ]
            .date()
        )

    def seed(
        self,
        seed,
    ):
        seed = int(
            seed
        )

        self._rng = (
            np.random.default_rng(
                seed
            )
        )

        return [
            seed
        ]

    def render(
        self,
        mode="human",
    ):
        if mode not in (
            "human",
            None,
        ):
            raise NotImplementedError(
                "cascade_reservoir "
                "only supports "
                "human rendering"
            )

        levels = {
            reservoir_id:
            float(
                self.physics[
                    reservoir_id
                ].level_from_storage(
                    self.storage_m3[
                        reservoir_id
                    ]
                )
            )
            for reservoir_id
            in self.reservoir_order
        }

        values = " | ".join(
            (
                f"{reservoir_id}: "
                f"Z={levels[reservoir_id]:.3f} m"
            )
            for reservoir_id
            in self.reservoir_order
        )

        print(
            f"{self.current_date} | "
            f"{values}"
        )

    def close(
        self,
    ):
        return None

    def _build_base_observations(
        self,
    ):
        current_date = (
            self.current_date
        )

        stage = (
            self._resolve_operation_stage(
                current_date
            )
        )

        forcing = (
            self._get_current_forcing()
        )

        levels = {
            reservoir_id:
            float(
                self.physics[
                    reservoir_id
                ].level_from_storage(
                    self.storage_m3[
                        reservoir_id
                    ]
                )
            )
            for reservoir_id
            in self.reservoir_order
        }

        day_of_year = (
            current_date
            .timetuple()
            .tm_yday
        )

        year_length = (
            366.0
            if self._is_leap_year(
                current_date.year
            )
            else 365.0
        )

        phase = (
            2.0
            * math.pi
            * (
                day_of_year - 1
            )
            / year_length
        )

        season_sin = (
            math.sin(
                phase
            )
        )

        season_cos = (
            math.cos(
                phase
            )
        )

        total_storage_min = sum(
            self.min_storage_m3.values()
        )

        total_storage_max = sum(
            self.max_storage_m3.values()
        )

        total_storage = sum(
            self.storage_m3.values()
        )

        system_storage_norm = (
            self._normalize(
                total_storage,
                total_storage_min,
                total_storage_max,
            )
        )

        total_previous_power = sum(
            (
                0.0
                if self.previous_power_mw[
                    reservoir_id
                ]
                is None
                else float(
                    self.previous_power_mw[
                        reservoir_id
                    ]
                )
            )
            for reservoir_id
            in self.reservoir_order
        )

        system_power_norm = (
            total_previous_power
            / self.total_installed_capacity_mw
        )

        flood_flag = (
            1.0
            if stage
            == "flood_control"
            else 0.0
        )

        result = {}

        for index, reservoir_id in enumerate(
            self.reservoir_order
        ):
            mapper = (
                self.action_mappers[
                    reservoir_id
                ]
            )

            current_level = (
                levels[
                    reservoir_id
                ]
            )

            p2 = (
                self._get_p2_constraints(
                    reservoir_id,
                    stage,
                )
            )

            p1_min = (
                self.hard_min_level_m[
                    reservoir_id
                ]
            )

            p1_max = (
                self.safety_upper_level_m[
                    reservoir_id
                ]
            )

            if (
                reservoir_id
                == "WDD"
            ):
                local_forcing = (
                    forcing[
                        "external_inflow_m3s"
                    ]
                )

            else:
                # Direct upstream current-day
                # release is not known yet.
                # Only interval inflow is known
                # in the base observation.
                local_forcing = (
                    forcing[
                        "interval_inflows_m3s"
                    ][
                        reservoir_id
                    ]
                )

            previous_action = (
                self.previous_action[
                    reservoir_id
                ]
            )

            previous_release = (
                self.previous_release_m3s[
                    reservoir_id
                ]
            )

            previous_power = (
                self.previous_power_mw[
                    reservoir_id
                ]
            )

            local_features = [
                self._normalize(
                    current_level,
                    p1_min,
                    p1_max,
                ),
                self._normalize(
                    self.storage_m3[
                        reservoir_id
                    ],
                    self.min_storage_m3[
                        reservoir_id
                    ],
                    self.max_storage_m3[
                        reservoir_id
                    ],
                ),
                (
                    float(
                        local_forcing
                    )
                    / mapper
                    .map_max_release_m3s
                ),
                season_sin,
                season_cos,
                self._normalize(
                    p2.level.min_level_m,
                    p1_min,
                    p1_max,
                ),
                self._normalize(
                    p2.level.max_level_m,
                    p1_min,
                    p1_max,
                ),
                (
                    0.0
                    if previous_action
                    is None
                    else mapper
                    .normalize_action(
                        previous_action
                    )
                ),
                (
                    0.0
                    if previous_release
                    is None
                    else (
                        float(
                            previous_release
                        )
                        / mapper
                        .map_max_release_m3s
                    )
                ),
                (
                    0.0
                    if previous_power
                    is None
                    else (
                        float(
                            previous_power
                        )
                        / self
                        .installed_capacity_mw[
                            reservoir_id
                        ]
                    )
                ),
            ]

            if index == 0:
                upstream_features = [
                    0.0,
                    0.0,
                ]

            else:
                upstream_id = (
                    self.reservoir_order[
                        index - 1
                    ]
                )

                upstream_previous_release = (
                    self.previous_release_m3s[
                        upstream_id
                    ]
                )

                upstream_features = [
                    self._normalize(
                        levels[
                            upstream_id
                        ],
                        self.hard_min_level_m[
                            upstream_id
                        ],
                        self
                        .safety_upper_level_m[
                            upstream_id
                        ],
                    ),
                    (
                        0.0
                        if upstream_previous_release
                        is None
                        else (
                            float(
                                upstream_previous_release
                            )
                            / self
                            .max_total_release_m3s[
                                upstream_id
                            ]
                        )
                    ),
                ]

            if (
                index
                == self.n_agents - 1
            ):
                downstream_features = [
                    0.0,
                    0.0,
                ]

            else:
                downstream_id = (
                    self.reservoir_order[
                        index + 1
                    ]
                )

                remaining_storage = (
                    self.max_storage_m3[
                        downstream_id
                    ]
                    - self.storage_m3[
                        downstream_id
                    ]
                )

                downstream_range = (
                    self.max_storage_m3[
                        downstream_id
                    ]
                    - self.min_storage_m3[
                        downstream_id
                    ]
                )

                downstream_features = [
                    self._normalize(
                        levels[
                            downstream_id
                        ],
                        self.hard_min_level_m[
                            downstream_id
                        ],
                        self
                        .safety_upper_level_m[
                            downstream_id
                        ],
                    ),
                    (
                        remaining_storage
                        / downstream_range
                    ),
                ]

            system_features = [
                system_storage_norm,
                system_power_norm,
                flood_flag,
            ]

            observation = np.asarray(
                (
                    local_features
                    + upstream_features
                    + downstream_features
                    + system_features
                ),
                dtype=np.float32,
            )

            if (
                observation.shape
                != (
                    _BASE_OBS_DIM,
                )
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "base observation "
                    "dimension must be 17"
                )

            if not np.all(
                np.isfinite(
                    observation
                )
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "base observation contains "
                    "non-finite values"
                )

            result[
                reservoir_id
            ] = observation

        return result

    def _build_global_state(
        self,
        base_observations,
    ):
        state = np.concatenate(
            [
                base_observations[
                    reservoir_id
                ]
                for reservoir_id
                in self.reservoir_order
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        if (
            state.shape
            != (
                _SHARE_OBS_DIM,
            )
        ):
            raise RuntimeError(
                "global state dimension "
                "must be 85"
            )

        if not np.all(
            np.isfinite(
                state
            )
        ):
            raise RuntimeError(
                "global state contains "
                "non-finite values"
            )

        return state

    def _build_share_observations(
        self,
    ):
        base = (
            self._build_base_observations()
        )

        state = (
            self._build_global_state(
                base
            )
        )

        return [
            state.copy()
            for _ in range(
                self.n_agents
            )
        ]

    def _build_interface_observations(
        self,
    ):
        """Return homogeneous 18-D placeholders for ShareDummyVecEnv.

        These are not behavior-policy observations.

        The custom runner stores the real WDD 17-D observation
        and four downstream 18-D decision observations produced
        during S4 sequential sampling.
        """

        base = (
            self._build_base_observations()
        )

        return [
            np.concatenate(
                (
                    base[
                        reservoir_id
                    ],
                    np.zeros(
                        1,
                        dtype=np.float32,
                    ),
                )
            ).astype(
                np.float32,
                copy=False,
            )
            for reservoir_id
            in self.reservoir_order
        ]

    def _get_current_forcing(
        self,
    ):
        row = (
            self.daily_inflow.iloc[
                self._date_index
            ]
        )

        external = (
            _nonnegative_finite(
                row[
                    _INFLOW_COLUMNS[
                        "WDD"
                    ]
                ],
                (
                    "WDD_external_"
                    "inflow_m3s"
                ),
            )
        )

        intervals = {
            reservoir_id:
            _nonnegative_finite(
                row[
                    _INFLOW_COLUMNS[
                        reservoir_id
                    ]
                ],
                (
                    f"{reservoir_id}_"
                    "interval_inflow_m3s"
                ),
            )
            for reservoir_id
            in _DOWNSTREAM_ORDER
        }

        return {
            "external_inflow_m3s": (
                external
            ),
            "interval_inflows_m3s": (
                intervals
            ),
        }

    def _get_p2_constraints(
        self,
        reservoir_id,
        operation_stage,
    ):
        if (
            operation_stage
            not in self.level_control_config
        ):
            raise KeyError(
                "level_control does not "
                f"define {operation_stage}"
            )

        upper_column = (
            self.level_control_config[
                operation_stage
            ]
        )

        row = (
            self.level_limits[
                reservoir_id
            ]
        )

        if upper_column not in row:
            raise KeyError(
                f"{reservoir_id}: "
                "level limit column "
                f"{upper_column} "
                "does not exist"
            )

        upper_level = (
            _finite(
                row[
                    upper_column
                ],
                (
                    f"{reservoir_id}."
                    f"{upper_column}"
                ),
            )
        )

        lower_level = (
            self.hard_min_level_m[
                reservoir_id
            ]
        )

        p1_upper = (
            self.safety_upper_level_m[
                reservoir_id
            ]
        )

        if not (
            lower_level
            <= upper_level
            <= p1_upper
        ):
            raise ValueError(
                f"{reservoir_id}: "
                "invalid P2 "
                "level boundary"
            )

        return P2Constraints(
            level=LevelBounds(
                min_level_m=(
                    lower_level
                ),
                max_level_m=(
                    upper_level
                ),
            )
        )

    def _resolve_operation_stage(
        self,
        current_date,
    ):
        month_day = (
            current_date.strftime(
                "%m-%d"
            )
        )

        matched = []

        for stage, bounds in (
            self
            .operation_stage_config
            .items()
        ):
            if (
                not isinstance(
                    bounds,
                    Sequence,
                )
                or isinstance(
                    bounds,
                    (str, bytes),
                )
                or len(
                    bounds
                )
                != 2
            ):
                raise ValueError(
                    "operation stage "
                    f"{stage} must contain "
                    "[start_mm-dd, "
                    "end_mm-dd]"
                )

            start = str(
                bounds[
                    0
                ]
            )

            end = str(
                bounds[
                    1
                ]
            )

            if start <= end:
                active = (
                    start
                    <= month_day
                    <= end
                )

            else:
                active = (
                    month_day
                    >= start
                    or month_day
                    <= end
                )

            if active:
                matched.append(
                    str(
                        stage
                    )
                )

        if len(
            matched
        ) != 1:
            raise ValueError(
                f"{current_date}: "
                "expected exactly one "
                "operation stage, "
                f"received {matched}"
            )

        return matched[
            0
        ]

    def _parse_actions(
        self,
        actions,
    ):
        array = np.asarray(
            actions
        )

        if (
            array.shape
            == (
                self.n_agents,
                1,
            )
        ):
            array = array[
                :,
                0
            ]

        elif (
            array.shape
            != (
                self.n_agents,
            )
        ):
            raise ValueError(
                "actions must have "
                "shape (5,) or (5, 1)"
            )

        result = {}

        for index, reservoir_id in enumerate(
            self.reservoir_order
        ):
            value = array[
                index
            ]

            if isinstance(
                value,
                (bool, np.bool_),
            ):
                raise TypeError(
                    "action indices "
                    "cannot be boolean"
                )

            numeric = float(
                value
            )

            if (
                not math.isfinite(
                    numeric
                )
                or not numeric.is_integer()
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "action must be "
                    "an integer index"
                )

            action = int(
                numeric
            )

            num_actions = (
                self.action_mappers[
                    reservoir_id
                ].num_actions
            )

            if not (
                0
                <= action
                < num_actions
            ):
                raise IndexError(
                    f"{reservoir_id}: "
                    "action index "
                    "out of range"
                )

            result[
                reservoir_id
            ] = action

        return result

    def _validate_safe_joint_action(
        self,
        action_indices,
        safety_result,
    ) -> None:
        """Verify the complete selected S4 path before execution."""

        if not isinstance(
            safety_result,
            SafetyRecoveryResult,
        ):
            raise TypeError(
                "safety_result must be "
                "SafetyRecoveryResult"
            )

        upstream_action = None

        for index, reservoir_id in enumerate(
            self.reservoir_order
        ):
            if index == 0:
                mask = (
                    safety_result
                    .get_action_mask(
                        reservoir_id
                    )
                )

            else:
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

            expected_shape = (
                self.action_mappers[
                    reservoir_id
                ].num_actions,
            )

            if (
                mask.shape
                != expected_shape
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "safety mask has "
                    f"shape {mask.shape}, "
                    "expected "
                    f"{expected_shape}"
                )

            if not np.all(
                (
                    mask == 0
                )
                | (
                    mask == 1
                )
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "safety mask "
                    "must be binary"
                )

            if not np.any(
                mask
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    "empty action mask "
                    "reached step()"
                )

            action = (
                action_indices[
                    reservoir_id
                ]
            )

            if not bool(
                mask[
                    action
                ]
            ):
                raise RuntimeError(
                    f"{reservoir_id}: "
                    f"selected action {action} "
                    "is outside the S2/S3 "
                    "executable set"
                )

            upstream_action = (
                action
            )

    def _validate_transition_p1(
        self,
        reservoir_id,
        release_m3s,
        transition,
    ) -> None:
        p1 = (
            self.p1_constraints[
                reservoir_id
            ]
        )

        min_release = float(
            p1.release
            .min_release_m3s
        )

        max_release = float(
            p1.release
            .max_release_m3s
        )

        if (
            release_m3s
            < (
                min_release
                - _FLOW_TOLERANCE_M3S
            )
            or release_m3s
            > (
                max_release
                + _FLOW_TOLERANCE_M3S
            )
        ):
            raise RuntimeError(
                f"{reservoir_id}: "
                "executed release "
                "violates P1"
            )

        next_level = float(
            transition.next_level_m
        )

        min_level = float(
            p1.level.min_level_m
        )

        max_level = float(
            p1.level.max_level_m
        )

        if (
            next_level
            < (
                min_level
                - _LEVEL_TOLERANCE_M
            )
            or next_level
            > (
                max_level
                + _LEVEL_TOLERANCE_M
            )
        ):
            raise RuntimeError(
                f"{reservoir_id}: "
                "next level violates P1; "
                f"level={next_level}"
            )

        next_storage = float(
            transition.next_storage_m3
        )

        if (
            next_storage
            < (
                self.min_storage_m3[
                    reservoir_id
                ]
                - _STORAGE_TOLERANCE_M3
            )
            or next_storage
            > (
                self.max_storage_m3[
                    reservoir_id
                ]
                + _STORAGE_TOLERANCE_M3
            )
        ):
            raise RuntimeError(
                f"{reservoir_id}: "
                "next storage "
                "violates P1"
            )

    def _build_infos(
        self,
        *,
        executed_date,
        operation_stage,
        action_indices,
        actual_inflows,
        interval_inflows,
        target_releases,
        executed_releases,
        transitions,
        system_energy_mwh,
        reward,
        safety_result,
        constraint_result,
    ):
        p4_ratio = (
            float(
                "nan"
            )
            if (
                safety_result
                .p4_release_inflow_ratio
                is None
            )
            else float(
                safety_result
                .p4_release_inflow_ratio
            )
        )

        p3_fraction = float(
            safety_result
            .p3_relaxation_fraction
        )

        p2_violation = float(
            safety_result
            .p2_total_violation
        )

        forced_joint_action = (
            None
            if (
                safety_result
                .forced_joint_action
                is None
            )
            else [
                int(
                    value
                )
                for value
                in safety_result
                .forced_joint_action
            ]
        )

        constraint_ids = tuple(
            constraint_result
            .constraint_ids
        )

        raw_violations = np.asarray(
            constraint_result
            .raw_violations,
            dtype=np.float32,
        )

        normalized_violations = (
            np.asarray(
                constraint_result
                .normalized_violations,
                dtype=np.float32,
            )
        )

        constraint_costs = np.asarray(
            constraint_result.costs,
            dtype=np.float32,
        )

        active_flags = np.asarray(
            constraint_result
            .active_flags,
            dtype=np.float32,
        )

        budgets = np.asarray(
            constraint_result.budgets,
            dtype=np.float32,
        )

        infos = []

        for reservoir_id in (
            self.reservoir_order
        ):
            transition = (
                transitions[
                    reservoir_id
                ]
            )

            infos.append(
                {
                    "date": str(
                        executed_date
                    ),
                    "executed_date": str(
                        executed_date
                    ),
                    "reservoir_id": (
                        reservoir_id
                    ),
                    "operation_stage": (
                        operation_stage
                    ),
                    "action_index": int(
                        action_indices[
                            reservoir_id
                        ]
                    ),
                    "inflow_m3s": float(
                        actual_inflows[
                            reservoir_id
                        ]
                    ),
                    "interval_inflow_m3s": (
                        0.0
                        if (
                            reservoir_id
                            == "WDD"
                        )
                        else float(
                            interval_inflows[
                                reservoir_id
                            ]
                        )
                    ),
                    "target_release_m3s": float(
                        target_releases[
                            reservoir_id
                        ]
                    ),
                    "executed_release_m3s": float(
                        executed_releases[
                            reservoir_id
                        ]
                    ),
                    "release_m3s": float(
                        executed_releases[
                            reservoir_id
                        ]
                    ),
                    "turbine_release_m3s": float(
                        transition
                        .turbine_release_m3s
                    ),
                    "non_generation_release_m3s": float(
                        transition
                        .non_generation_release_m3s
                    ),
                    "next_storage_m3": float(
                        transition
                        .next_storage_m3
                    ),
                    "next_level_m": float(
                        transition
                        .next_level_m
                    ),
                    "power_mw": float(
                        transition.power_mw
                    ),
                    "energy_mwh": float(
                        transition.energy_mwh
                    ),
                    "system_energy_mwh": float(
                        system_energy_mwh
                    ),
                    "system_reward": float(
                        reward
                    ),
                    "safety_mode": str(
                        safety_result.mode
                    ),
                    "p4_release_inflow_ratio": (
                        p4_ratio
                    ),
                    "p3_relaxation_fraction": (
                        p3_fraction
                    ),
                    "p2_total_violation": (
                        p2_violation
                    ),
                    "forced_joint_action": (
                        forced_joint_action
                    ),
                    "constraint_ids": (
                        constraint_ids
                    ),
                    "constraint_raw_violations": (
                        raw_violations.copy()
                    ),
                    "constraint_normalized_violations": (
                        normalized_violations.copy()
                    ),
                    "constraint_costs": (
                        constraint_costs.copy()
                    ),
                    "constraint_active_flags": (
                        active_flags.copy()
                    ),
                    "constraint_budgets": (
                        budgets.copy()
                    ),
                    "bad_transition": False,
                }
            )

        return infos

    def _copy_prepared_context(
        self,
        context,
    ):
        return {
            "date": str(
                context[
                    "date"
                ]
            ),
            "operation_stage": str(
                context[
                    "operation_stage"
                ]
            ),
            "base_observations": {
                reservoir_id:
                np.asarray(
                    context[
                        "base_observations"
                    ][
                        reservoir_id
                    ],
                    dtype=np.float32,
                ).copy()
                for reservoir_id
                in self.reservoir_order
            },
            "share_observation": (
                np.asarray(
                    context[
                        "share_observation"
                    ],
                    dtype=np.float32,
                ).copy()
            ),
            "external_inflow_m3s": float(
                context[
                    "external_inflow_m3s"
                ]
            ),
            "interval_inflows_m3s": {
                reservoir_id:
                float(
                    context[
                        "interval_inflows_m3s"
                    ][
                        reservoir_id
                    ]
                )
                for reservoir_id
                in _DOWNSTREAM_ORDER
            },
            # SafetyRecoveryResult is frozen and
            # contains readonly S203 masks.
            "safety_result": (
                context[
                    "safety_result"
                ]
            ),
        }

    def _get_initial_storage(
        self,
        reservoir_id,
    ):
        source = (
            self.data_loader
            .initial_storage_m3
        )

        if isinstance(
            source,
            Mapping,
        ):
            value = source[
                reservoir_id
            ]

        elif isinstance(
            source,
            pd.Series,
        ):
            value = source.loc[
                reservoir_id
            ]

        else:
            raise TypeError(
                "initial_storage_m3 "
                "must be a mapping "
                "or Series"
            )

        return _positive_finite(
            value,
            (
                f"{reservoir_id}."
                "initial_storage_m3"
            ),
        )

    def _get_level_limit_row(
        self,
        reservoir_id,
    ):
        source = (
            self.data_loader
            .level_limits
        )

        if isinstance(
            source,
            pd.DataFrame,
        ):
            if (
                "reservoir_id"
                in source.columns
            ):
                matches = source[
                    source[
                        "reservoir_id"
                    ].astype(
                        str
                    )
                    == reservoir_id
                ]

                if len(
                    matches
                ) != 1:
                    raise ValueError(
                        f"{reservoir_id}: "
                        "expected one "
                        "level-limit row"
                    )

                row = (
                    matches.iloc[
                        0
                    ].to_dict()
                )

            else:
                if (
                    reservoir_id
                    not in source.index
                ):
                    raise KeyError(
                        f"{reservoir_id}: "
                        "missing "
                        "level-limit row"
                    )

                row = (
                    source.loc[
                        reservoir_id
                    ].to_dict()
                )

        elif isinstance(
            source,
            Mapping,
        ):
            row = dict(
                source[
                    reservoir_id
                ]
            )

        else:
            raise TypeError(
                "level_limits must be "
                "a DataFrame or mapping"
            )

        required = (
            "hard_min_level_m",
            "normal_upper_level_m",
            "flood_limit_level_m",
            "safety_upper_level_m",
        )

        for key in required:
            if key not in row:
                raise KeyError(
                    f"{reservoir_id}: "
                    "missing level limit "
                    f"{key}"
                )

            row[
                key
            ] = _finite(
                row[
                    key
                ],
                (
                    f"{reservoir_id}."
                    f"{key}"
                ),
            )

        return row

    @staticmethod
    def _normalize(
        value,
        lower,
        upper,
    ):
        value = float(
            value
        )

        lower = float(
            lower
        )

        upper = float(
            upper
        )

        denominator = (
            upper - lower
        )

        if denominator <= 0.0:
            raise ValueError(
                "normalization interval "
                "must be positive"
            )

        return (
            value - lower
        ) / denominator

    @staticmethod
    def _is_leap_year(
        year,
    ):
        return (
            year % 4 == 0
            and (
                year % 100 != 0
                or year % 400 == 0
            )
        )


def _finite(
    value,
    name: str,
) -> float:
    if value is None:
        raise ValueError(
            f"{name} cannot be None"
        )

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
