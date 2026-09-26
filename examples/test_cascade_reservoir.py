from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


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

_INTERFACE_OBS_DIM = 18
_BASE_OBS_DIM = 17
_SHARE_OBS_DIM = 85


def _ensure_harl_importable() -> Path:
    """Make the repository root importable when run from examples/."""

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    if not (repo_root / "harl").is_dir():
        raise RuntimeError(
            "Cannot locate the HARL repository root. "
            "Place this file at HARL/examples/test_cascade_reservoir.py "
            "and run it from the HARL project."
        )

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    return repo_root


def _assert_finite(name: str, values) -> np.ndarray:
    array = np.asarray(values)

    if not np.all(np.isfinite(array)):
        raise AssertionError(
            f"{name} contains non-finite values"
        )

    return array


def _assert_binary(name: str, values) -> np.ndarray:
    array = np.asarray(values)

    if not np.all(
        (array == 0)
        | (array == 1)
    ):
        raise AssertionError(
            f"{name} must contain only 0/1 values"
        )

    return array


def _print_pass(message: str) -> None:
    print(f"[PASS] {message}")


def test_reset_and_prepare_step() -> None:
    """Step 1 smoke test: reset -> prepare_step only."""

    repo_root = _ensure_harl_importable()

    from harl.envs.cascade_reservoir.cascade_reservoir_env import (
        CascadeReservoirEnv,
    )
    from harl.envs.cascade_reservoir.s3_relaxation import (
        SafetyRecoveryResult,
    )
    from harl.utils.configs_tools import (
        get_defaults_yaml_args,
    )

    print("=" * 72)
    print("Cascade reservoir smoke test - STEP 1")
    print("Scope: reset() -> prepare_step()")
    print(f"Repository: {repo_root}")
    print("=" * 72)

    _, env_args = get_defaults_yaml_args(
        "happo",
        "cascade_reservoir",
    )

    env = CascadeReservoirEnv(
        env_args
    )

    try:
        # ======================================================
        # A. reset()
        # ======================================================
        (
            observations,
            share_observations,
            available_actions,
        ) = env.reset()

        if env.n_agents != 5:
            raise AssertionError(
                f"Expected 5 agents, got {env.n_agents}"
            )

        if tuple(env.reservoir_order) != _RESERVOIR_ORDER:
            raise AssertionError(
                "Reservoir order must be "
                "WDD -> BHT -> XLD -> XJB -> THR"
            )

        _print_pass(
            "environment has five reservoirs in hydraulic order"
        )

        if len(observations) != 5:
            raise AssertionError(
                "reset() must return five interface observations"
            )

        for reservoir_id, observation in zip(
            _RESERVOIR_ORDER,
            observations,
        ):
            observation = _assert_finite(
                f"{reservoir_id} reset observation",
                observation,
            )

            if observation.shape != (
                _INTERFACE_OBS_DIM,
            ):
                raise AssertionError(
                    f"{reservoir_id} reset interface observation "
                    f"must have shape ({_INTERFACE_OBS_DIM},), "
                    f"got {observation.shape}"
                )

        _print_pass(
            "reset interface observations are finite and uniformly 18-D"
        )

        if len(share_observations) != 5:
            raise AssertionError(
                "reset() must return five shared observations"
            )

        reference_share = None

        for reservoir_id, share_obs in zip(
            _RESERVOIR_ORDER,
            share_observations,
        ):
            share_obs = _assert_finite(
                f"{reservoir_id} share observation",
                share_obs,
            ).astype(
                np.float32,
                copy=False,
            )

            if share_obs.shape != (
                _SHARE_OBS_DIM,
            ):
                raise AssertionError(
                    f"{reservoir_id} shared observation "
                    f"must have shape ({_SHARE_OBS_DIM},), "
                    f"got {share_obs.shape}"
                )

            if reference_share is None:
                reference_share = share_obs.copy()

            elif not np.array_equal(
                share_obs,
                reference_share,
            ):
                raise AssertionError(
                    "EP shared observations must be identical "
                    "for all five agents"
                )

        _print_pass(
            "reset EP shared state is finite, 85-D, and identical across agents"
        )

        action_mappers = (
            env.get_action_mappers()
        )

        if set(action_mappers) != set(
            _RESERVOIR_ORDER
        ):
            raise AssertionError(
                "get_action_mappers() must expose exactly five reservoirs"
            )

        if len(available_actions) != 5:
            raise AssertionError(
                "reset() must return five available-action vectors"
            )

        for reservoir_id, mask in zip(
            _RESERVOIR_ORDER,
            available_actions,
        ):
            mapper = action_mappers[
                reservoir_id
            ]

            if str(
                mapper.reservoir_id
            ) != reservoir_id:
                raise AssertionError(
                    f"{reservoir_id}: mapper reservoir_id mismatch"
                )

            mask = _assert_binary(
                f"{reservoir_id} reset available_actions",
                mask,
            )

            if mask.shape != (
                mapper.num_actions,
            ):
                raise AssertionError(
                    f"{reservoir_id}: reset available_actions shape "
                    "does not match mapper.num_actions"
                )

            if not np.all(
                mask == 1
            ):
                raise AssertionError(
                    f"{reservoir_id}: reset available_actions "
                    "must be the HARL all-one placeholder"
                )

        _print_pass(
            "reset available-actions are all-one HARL interface placeholders"
        )

        # ======================================================
        # B. prepare_step() must be read-only.
        # ======================================================
        date_before = env.current_date

        storage_before = {
            reservoir_id:
            float(
                env.storage_m3[
                    reservoir_id
                ]
            )
            for reservoir_id
            in _RESERVOIR_ORDER
        }

        previous_release_before = {
            reservoir_id:
            env.previous_release_m3s[
                reservoir_id
            ]
            for reservoir_id
            in _RESERVOIR_ORDER
        }

        context = env.prepare_step()

        required_keys = {
            "date",
            "operation_stage",
            "base_observations",
            "share_observation",
            "external_inflow_m3s",
            "interval_inflows_m3s",
            "safety_result",
        }

        missing = (
            required_keys
            - set(context)
        )

        if missing:
            raise AssertionError(
                f"prepare_step() missing keys: {missing}"
            )

        _print_pass(
            "prepare_step() returned the complete S1-S3 context"
        )

        if env.current_date != date_before:
            raise AssertionError(
                "prepare_step() changed the simulation date"
            )

        for reservoir_id in _RESERVOIR_ORDER:
            if not np.isclose(
                env.storage_m3[
                    reservoir_id
                ],
                storage_before[
                    reservoir_id
                ],
                rtol=0.0,
                atol=0.0,
            ):
                raise AssertionError(
                    f"prepare_step() changed {reservoir_id} storage"
                )

            if (
                env.previous_release_m3s[
                    reservoir_id
                ]
                != previous_release_before[
                    reservoir_id
                ]
            ):
                raise AssertionError(
                    f"prepare_step() changed "
                    f"{reservoir_id} previous release"
                )

        _print_pass(
            "prepare_step() is read-only: date/storage/history did not change"
        )

        # ======================================================
        # C. Base observations and centralized state.
        # ======================================================
        base_observations = (
            context[
                "base_observations"
            ]
        )

        if set(
            base_observations
        ) != set(
            _RESERVOIR_ORDER
        ):
            raise AssertionError(
                "base_observations must contain exactly five reservoirs"
            )

        ordered_base = []

        for reservoir_id in (
            _RESERVOIR_ORDER
        ):
            observation = _assert_finite(
                f"{reservoir_id} base observation",
                base_observations[
                    reservoir_id
                ],
            ).astype(
                np.float32,
                copy=False,
            )

            if observation.shape != (
                1,
                _BASE_OBS_DIM,
            ):
                raise AssertionError(
                    f"{reservoir_id}: base observation must have "
                    f"shape (1, {_BASE_OBS_DIM}), got {observation.shape}"
                )

            ordered_base.append(
                observation
            )

        share_observation = _assert_finite(
            "prepare_step share_observation",
            context[
                "share_observation"
            ],
        ).astype(
            np.float32,
            copy=False,
        )

        if share_observation.shape != (
            1,
            _SHARE_OBS_DIM,
        ):
            raise AssertionError(
                "prepare_step share_observation must "
                f"have shape (1, {_SHARE_OBS_DIM})"
            )

        reconstructed_share = (
            np.concatenate(
                ordered_base,
                axis=1,
            )
        )

        if not np.array_equal(
            reconstructed_share,
            share_observation,
        ):
            raise AssertionError(
                "share_observation does not equal the concatenation "
                "of the five base observations"
            )

        _print_pass(
            "five 17-D base observations reconstruct the 85-D centralized state"
        )

        # ======================================================
        # D. Current forcing.
        # ======================================================
        external_inflow = float(
            context[
                "external_inflow_m3s"
            ]
        )

        if (
            not np.isfinite(
                external_inflow
            )
            or external_inflow < 0.0
        ):
            raise AssertionError(
                "WDD external inflow must be finite and nonnegative"
            )

        interval_inflows = (
            context[
                "interval_inflows_m3s"
            ]
        )

        if set(
            interval_inflows
        ) != set(
            _DOWNSTREAM_ORDER
        ):
            raise AssertionError(
                "interval_inflows_m3s must contain "
                "BHT, XLD, XJB and THR only"
            )

        for reservoir_id in (
            _DOWNSTREAM_ORDER
        ):
            value = float(
                interval_inflows[
                    reservoir_id
                ]
            )

            if (
                not np.isfinite(
                    value
                )
                or value < 0.0
            ):
                raise AssertionError(
                    f"{reservoir_id}: interval inflow "
                    "must be finite and nonnegative"
                )

        _print_pass(
            "WDD external inflow and four downstream interval inflows are valid"
        )

        # ======================================================
        # E. S2/S3 safety result.
        # ======================================================
        safety_result = (
            context[
                "safety_result"
            ]
        )

        if not isinstance(
            safety_result,
            SafetyRecoveryResult,
        ):
            raise AssertionError(
                "prepare_step safety_result must be SafetyRecoveryResult"
            )

        expected_num_actions = tuple(
            action_mappers[
                reservoir_id
            ].num_actions
            for reservoir_id
            in _RESERVOIR_ORDER
        )

        if tuple(
            safety_result.num_actions
        ) != expected_num_actions:
            raise AssertionError(
                "SafetyRecoveryResult action dimensions do not "
                "match the five action mappers"
            )

        if not (
            safety_result
            .extendability
            .has_joint_feasible_action
        ):
            raise AssertionError(
                "prepare_step produced no complete P1-safe "
                "cascade action path"
            )

        wdd_mask = _assert_binary(
            "WDD S2/S3 action mask",
            safety_result
            .get_action_mask(
                "WDD"
            ),
        )

        if wdd_mask.shape != (
            expected_num_actions[
                0
            ],
        ):
            raise AssertionError(
                "WDD safety mask has wrong action dimension"
            )

        if not np.any(
            wdd_mask
        ):
            raise AssertionError(
                "WDD safety mask is empty"
            )

        _print_pass(
            "S2/S3 returned a valid non-empty cascade-feasible WDD mask"
        )

        # ======================================================
        # F. prepare_step() defensive-copy contract.
        # ======================================================
        original_first_value = float(
            context[
                "base_observations"
            ][
                "WDD"
            ][
                0,
                0,
            ]
        )

        context[
            "base_observations"
        ][
            "WDD"
        ][
            0,
            0,
        ] = (
            original_first_value
            + 12345.0
        )

        repeated_context = (
            env.prepare_step()
        )

        repeated_first_value = float(
            repeated_context[
                "base_observations"
            ][
                "WDD"
            ][
                0,
                0,
            ]
        )

        if not np.isclose(
            repeated_first_value,
            original_first_value,
            rtol=0.0,
            atol=0.0,
        ):
            raise AssertionError(
                "prepare_step() leaked mutable observation arrays "
                "from its internal cached context"
            )

        if env.current_date != date_before:
            raise AssertionError(
                "second prepare_step() call changed the simulation date"
            )

        _print_pass(
            "prepare_step() returns defensive copies and remains idempotent"
        )

        print("-" * 72)
        print(
            "STEP 1 PASSED | "
            f"date={context['date']} | "
            f"stage={context['operation_stage']} | "
            f"safety_mode={safety_result.mode} | "
            f"WDD_feasible_actions={int(np.count_nonzero(wdd_mask))}"
        )
        print("-" * 72)

    finally:
        env.close()


def main() -> None:
    test_reset_and_prepare_step()


if __name__ == "__main__":
    main()
