from __future__ import annotations

import math


def compute_downstream_inflow(
    upstream_flow: float,
    interval_inflow: float,
) -> float:
    upstream_flow = _validate_flow(
        upstream_flow,
        "upstream_flow",
    )

    interval_inflow = _validate_flow(
        interval_inflow,
        "interval_inflow",
    )

    return upstream_flow + interval_inflow


def _validate_flow(
    value: float,
    name: str,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if value < 0.0:
        raise ValueError(
            f"{name} cannot be negative"
        )

    return value
