"""Runner registry for the cascade reservoir research pipeline."""
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

RUNNER_REGISTRY = {"happo": CascadeReservoirRunner}
