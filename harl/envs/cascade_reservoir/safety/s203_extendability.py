from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Sequence, Tuple

import numpy as np

from harl.envs.cascade_reservoir.safety.s202_compatibility import (
    CompatibilityResult,
)


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_EXPECTED_LINKS = (
    ("WDD", "BHT"),
    ("BHT", "XLD"),
    ("XLD", "XJB"),
    ("XJB", "THR"),
)


@dataclass(frozen=True)
class ExtendabilityResult:
    """Result of S203 downstream-extendability recursion.

    The four compatibility matrices correspond to:

        C_BHT : WDD -> BHT
        C_XLD : BHT -> XLD
        C_XJB : XLD -> XJB
        C_THR : XJB -> THR

    The four downstream-extendability masks correspond to:

        index 0:
            whether each WDD action can extend through
            BHT -> XLD -> XJB -> THR

        index 1:
            whether each BHT action can extend through
            XLD -> XJB -> THR

        index 2:
            whether each XLD action can extend through
            XJB -> THR

        index 3:
            whether each XJB action can extend to THR

    root_action_mask is the final WDD cascade-feasible mask:

        WDD local feasibility
        AND
        WDD downstream extendability.
    """

    reservoir_order: Tuple[str, ...]

    root_local_mask: np.ndarray

    compatibility_matrices: Tuple[
        np.ndarray,
        ...,
    ]

    downstream_extendability_masks: Tuple[
        np.ndarray,
        ...,
    ]

    root_action_mask: np.ndarray

    def __post_init__(self):
        reservoir_order = tuple(
            str(reservoir_id)
            for reservoir_id
            in self.reservoir_order
        )

        if reservoir_order != _RESERVOIR_ORDER:
            raise ValueError(
                "reservoir_order must be "
                "('WDD', 'BHT', 'XLD', 'XJB', 'THR')"
            )

        root_local_mask = _validate_binary_vector(
            self.root_local_mask,
            "root_local_mask",
        )

        compatibility_matrices = tuple(
            _validate_binary_matrix(
                matrix,
                (
                    f"compatibility_matrices"
                    f"[{index}]"
                ),
            )
            for index, matrix
            in enumerate(
                self.compatibility_matrices
            )
        )

        if (
            len(
                compatibility_matrices
            )
            != len(
                _EXPECTED_LINKS
            )
        ):
            raise ValueError(
                "exactly four compatibility "
                "matrices are required"
            )

        _validate_matrix_chain(
            compatibility_matrices
        )

        if (
            root_local_mask.size
            != compatibility_matrices[
                0
            ].shape[0]
        ):
            raise ValueError(
                "root_local_mask length does not "
                "match WDD action dimension"
            )

        downstream_masks = tuple(
            _validate_binary_vector(
                mask,
                (
                    "downstream_extendability_masks"
                    f"[{index}]"
                ),
            )
            for index, mask
            in enumerate(
                self.downstream_extendability_masks
            )
        )

        if (
            len(
                downstream_masks
            )
            != len(
                _EXPECTED_LINKS
            )
        ):
            raise ValueError(
                "exactly four downstream "
                "extendability masks are required"
            )

        for index, mask in enumerate(
            downstream_masks
        ):
            expected_size = (
                compatibility_matrices[
                    index
                ].shape[0]
            )

            if mask.size != expected_size:
                raise ValueError(
                    "downstream extendability mask "
                    f"{index} must have length "
                    f"{expected_size}, received "
                    f"{mask.size}"
                )

        # Verify that the stored recursion result is
        # mathematically consistent with the compatibility
        # matrices.
        expected_masks = (
            _compute_downstream_masks(
                compatibility_matrices
            )
        )

        for index, (
            actual_mask,
            expected_mask,
        ) in enumerate(
            zip(
                downstream_masks,
                expected_masks,
            )
        ):
            if not np.array_equal(
                actual_mask,
                expected_mask,
            ):
                raise ValueError(
                    "downstream extendability mask "
                    f"{index} is inconsistent with "
                    "the compatibility matrices"
                )

        root_action_mask = (
            _validate_binary_vector(
                self.root_action_mask,
                "root_action_mask",
            )
        )

        expected_root_mask = (
            root_local_mask
            & downstream_masks[0]
        )

        if not np.array_equal(
            root_action_mask,
            expected_root_mask,
        ):
            raise ValueError(
                "root_action_mask is inconsistent "
                "with root_local_mask and downstream "
                "extendability"
            )

        object.__setattr__(
            self,
            "reservoir_order",
            reservoir_order,
        )

        object.__setattr__(
            self,
            "root_local_mask",
            root_local_mask,
        )

        object.__setattr__(
            self,
            "compatibility_matrices",
            compatibility_matrices,
        )

        object.__setattr__(
            self,
            "downstream_extendability_masks",
            downstream_masks,
        )

        object.__setattr__(
            self,
            "root_action_mask",
            root_action_mask,
        )

    @property
    def has_joint_feasible_action(
        self,
    ) -> bool:
        """Whether at least one complete five-reservoir path exists."""

        return bool(
            np.any(
                self.root_action_mask
            )
        )

    @property
    def root_feasible_action_indices(
        self,
    ) -> Tuple[int, ...]:
        """Cascade-feasible WDD action indices."""

        return tuple(
            int(index)
            for index
            in np.flatnonzero(
                self.root_action_mask
            )
        )

    @property
    def action_counts(
        self,
    ) -> Tuple[int, ...]:
        """Number of candidate actions for WDD, BHT, XLD, XJB, THR."""

        matrices = (
            self.compatibility_matrices
        )

        return (
            int(
                matrices[0].shape[0]
            ),
            int(
                matrices[0].shape[1]
            ),
            int(
                matrices[1].shape[1]
            ),
            int(
                matrices[2].shape[1]
            ),
            int(
                matrices[3].shape[1]
            ),
        )

    def get_downstream_extendability_mask(
        self,
        reservoir_id: str,
    ) -> np.ndarray:
        """Return whether each action can still reach THR.

        This quantity exists only for WDD, BHT, XLD and XJB.

        THR is already the terminal reservoir, so asking for
        downstream extendability of THR has no meaning.
        """

        reservoir_index = (
            self._get_reservoir_index(
                reservoir_id
            )
        )

        if (
            reservoir_index
            == len(
                self.reservoir_order
            ) - 1
        ):
            raise ValueError(
                "THR has no downstream "
                "extendability mask"
            )

        return (
            self
            .downstream_extendability_masks[
                reservoir_index
            ]
            .copy()
        )

    def get_action_mask(
        self,
        reservoir_id: str,
        upstream_action_index: Optional[
            int
        ] = None,
    ) -> np.ndarray:
        """Return the S203 cascade-feasible action mask.

        WDD
        ---
        M_WDD =
            M_WDD_local
            AND
            b_BHT

        BHT / XLD / XJB
        ----------------
        Once the directly upstream action is fixed:

            M_i =
                C_i[a_upstream, :]
                AND
                b_next

        THR
        ---
        Since THR is the terminal reservoir:

            M_THR =
                C_THR[a_XJB, :]
        """

        reservoir_index = (
            self._get_reservoir_index(
                reservoir_id
            )
        )

        # WDD has no directly upstream reservoir.
        if reservoir_index == 0:
            if (
                upstream_action_index
                is not None
            ):
                raise ValueError(
                    "WDD does not accept "
                    "upstream_action_index"
                )

            return (
                self.root_action_mask.copy()
            )

        if upstream_action_index is None:
            raise ValueError(
                f"{reservoir_id} requires "
                "upstream_action_index"
            )

        compatibility_matrix = (
            self.compatibility_matrices[
                reservoir_index - 1
            ]
        )

        upstream_action_index = (
            _validate_action_index(
                upstream_action_index,
                compatibility_matrix.shape[
                    0
                ],
                "upstream_action_index",
            )
        )

        action_mask = (
            compatibility_matrix[
                upstream_action_index,
                :,
            ].copy()
        )

        # Middle reservoirs must not merely be compatible
        # with the already selected upstream action.
        #
        # They must also still have at least one complete
        # feasible continuation to THR.
        if (
            reservoir_index
            < len(
                self.reservoir_order
            ) - 1
        ):
            action_mask &= (
                self
                .downstream_extendability_masks[
                    reservoir_index
                ]
            )

        # THR is the terminal reservoir.  Its compatibility
        # row already contains its S201 local feasibility
        # under the candidate inflow associated with the
        # selected XJB action, so no additional tail mask is
        # needed.
        return action_mask

    def get_action_indices(
        self,
        reservoir_id: str,
        upstream_action_index: Optional[
            int
        ] = None,
    ) -> Tuple[int, ...]:
        """Return feasible action indices for one S4 decision."""

        mask = self.get_action_mask(
            reservoir_id=reservoir_id,
            upstream_action_index=(
                upstream_action_index
            ),
        )

        return tuple(
            int(index)
            for index
            in np.flatnonzero(
                mask
            )
        )

    def _get_reservoir_index(
        self,
        reservoir_id: str,
    ) -> int:
        reservoir_id = str(
            reservoir_id
        )

        try:
            return (
                self.reservoir_order.index(
                    reservoir_id
                )
            )
        except ValueError as exc:
            raise KeyError(
                "unknown reservoir_id: "
                f"{reservoir_id}"
            ) from exc


def compute_extendability(
    wdd_local_mask,
    compatibility_results: Sequence[
        CompatibilityResult
    ],
) -> ExtendabilityResult:
    """Run S203 backward downstream-extendability recursion.

    Parameters
    ----------
    wdd_local_mask:
        WDD S201 local feasible-action mask.

        With the current S201 implementation, the preferred input is:

            wdd_local_result.feasible_mask

    compatibility_results:
        Exactly four S202 results in hydraulic order:

            WDD -> BHT
            BHT -> XLD
            XLD -> XJB
            XJB -> THR

    Returns
    -------
    ExtendabilityResult
        Complete S203 recursion result and sequential S4 masks.

    Notes
    -----
    S203 performs no physical simulation and no constraint relaxation.
    It only propagates binary compatibility information backward.
    """

    compatibility_results = tuple(
        compatibility_results
    )

    if (
        len(
            compatibility_results
        )
        != len(
            _EXPECTED_LINKS
        )
    ):
        raise ValueError(
            "exactly four CompatibilityResult "
            "objects are required"
        )

    compatibility_matrices = []

    for index, (
        expected_upstream,
        expected_downstream,
    ) in enumerate(
        _EXPECTED_LINKS
    ):
        result = (
            compatibility_results[
                index
            ]
        )

        if not isinstance(
            result,
            CompatibilityResult,
        ):
            raise TypeError(
                "compatibility_results must "
                "contain CompatibilityResult "
                "objects"
            )

        if (
            result.upstream_reservoir_id
            != expected_upstream
            or result.downstream_reservoir_id
            != expected_downstream
        ):
            raise ValueError(
                f"compatibility_results[{index}] "
                "must represent "
                f"{expected_upstream} -> "
                f"{expected_downstream}, received "
                f"{result.upstream_reservoir_id} -> "
                f"{result.downstream_reservoir_id}"
            )

        matrix = (
            _validate_binary_matrix(
                result.compatibility_matrix,
                (
                    f"{expected_upstream}_"
                    f"{expected_downstream}"
                ),
            )
        )

        compatibility_matrices.append(
            matrix
        )

    compatibility_matrices = tuple(
        compatibility_matrices
    )

    _validate_matrix_chain(
        compatibility_matrices
    )

    wdd_local_mask = (
        _validate_binary_vector(
            wdd_local_mask,
            "wdd_local_mask",
        )
    )

    if (
        wdd_local_mask.size
        != compatibility_matrices[
            0
        ].shape[0]
    ):
        raise ValueError(
            "wdd_local_mask length does not "
            "match the WDD action dimension "
            "of the WDD -> BHT matrix"
        )

    downstream_masks = (
        _compute_downstream_masks(
            compatibility_matrices
        )
    )

    root_action_mask = (
        wdd_local_mask
        & downstream_masks[0]
    )

    root_action_mask = (
        root_action_mask.copy()
    )

    root_action_mask.setflags(
        write=False
    )

    return ExtendabilityResult(
        reservoir_order=(
            _RESERVOIR_ORDER
        ),
        root_local_mask=(
            wdd_local_mask
        ),
        compatibility_matrices=(
            compatibility_matrices
        ),
        downstream_extendability_masks=(
            downstream_masks
        ),
        root_action_mask=(
            root_action_mask
        ),
    )


def _compute_downstream_masks(
    compatibility_matrices: Tuple[
        np.ndarray,
        ...,
    ],
) -> Tuple[np.ndarray, ...]:
    """Compute b-masks from THR backward to WDD.

    For the last link:

        b_THR(a_XJB)
        =
        exists a_THR:
            C_THR(a_XJB, a_THR) = 1

    Then recursively:

        b_i(a_up)
        =
        exists a_i:
            C_i(a_up, a_i) = 1
            and
            b_next(a_i) = 1
    """

    num_links = len(
        compatibility_matrices
    )

    if num_links != len(
        _EXPECTED_LINKS
    ):
        raise ValueError(
            "_compute_downstream_masks "
            "requires four matrices"
        )

    masks = [
        None
    ] * num_links

    # XJB -> THR:
    #
    # An XJB action is downstream-extendable iff
    # at least one THR action is compatible.
    masks[-1] = np.any(
        compatibility_matrices[
            -1
        ],
        axis=1,
    )

    # XLD -> XJB,
    # BHT -> XLD,
    # WDD -> BHT.
    for link_index in range(
        num_links - 2,
        -1,
        -1,
    ):
        compatibility_matrix = (
            compatibility_matrices[
                link_index
            ]
        )

        next_extendable = (
            masks[
                link_index + 1
            ]
        )

        if (
            compatibility_matrix.shape[
                1
            ]
            != next_extendable.size
        ):
            raise ValueError(
                "compatibility matrix column "
                "dimension does not match the "
                "next downstream extendability "
                "mask"
            )

        masks[
            link_index
        ] = np.any(
            compatibility_matrix
            & next_extendable[
                np.newaxis,
                :,
            ],
            axis=1,
        )

    readonly_masks = []

    for mask in masks:
        mask = np.asarray(
            mask,
            dtype=bool,
        ).copy()

        mask.setflags(
            write=False
        )

        readonly_masks.append(
            mask
        )

    return tuple(
        readonly_masks
    )


def _validate_matrix_chain(
    matrices: Sequence[
        np.ndarray
    ],
) -> None:
    """Validate adjacent S202 action dimensions."""

    matrices = tuple(
        matrices
    )

    if (
        len(
            matrices
        )
        != len(
            _EXPECTED_LINKS
        )
    ):
        raise ValueError(
            "matrix chain must contain "
            "exactly four matrices"
        )

    for index in range(
        len(matrices) - 1
    ):
        current_matrix = (
            matrices[
                index
            ]
        )

        next_matrix = (
            matrices[
                index + 1
            ]
        )

        # Example:
        #
        # C_BHT columns represent BHT actions.
        # C_XLD rows also represent BHT actions.
        #
        # Their dimensions must therefore agree.
        if (
            current_matrix.shape[1]
            != next_matrix.shape[0]
        ):
            raise ValueError(
                "adjacent compatibility matrix "
                "dimensions do not match: "
                f"matrix {index} has "
                f"{current_matrix.shape[1]} "
                "downstream actions, but matrix "
                f"{index + 1} has "
                f"{next_matrix.shape[0]} "
                "upstream actions"
            )


def _validate_binary_vector(
    values,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional"
        )

    if array.size < 2:
        raise ValueError(
            f"{name} must contain at least "
            "two actions"
        )

    if array.dtype != np.bool_:
        if not np.all(
            (array == 0)
            | (array == 1)
        ):
            raise ValueError(
                f"{name} must contain "
                "only binary values"
            )

    array = array.astype(
        bool,
        copy=True,
    )

    array.setflags(
        write=False
    )

    return array


def _validate_binary_matrix(
    values,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional"
        )

    if (
        array.shape[0] < 2
        or array.shape[1] < 2
    ):
        raise ValueError(
            f"{name} must contain at least "
            "two actions on each side"
        )

    if array.dtype != np.bool_:
        if not np.all(
            (array == 0)
            | (array == 1)
        ):
            raise ValueError(
                f"{name} must contain "
                "only binary values"
            )

    array = array.astype(
        bool,
        copy=True,
    )

    array.setflags(
        write=False
    )

    return array


def _validate_action_index(
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