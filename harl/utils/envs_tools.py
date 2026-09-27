"""Tools for HARL."""

import os
import random

import numpy as np
import torch

from harl.envs.env_wrappers import (
    ShareDummyVecEnv,
)


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = (
        torch.from_numpy(value)
        if isinstance(value, np.ndarray)
        else value
    )
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.

    Args:
        obs_space: gym.spaces or list observation space.

    Returns:
        Observation shape.
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError

    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.

    Args:
        act_space: gym.spaces action space.

    Returns:
        Action shape.
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:
        raise NotImplementedError(
            f"unsupported action space: "
            f"{act_space.__class__.__name__}"
        )

    return act_shape


def _make_cascade_env(env_name, seed, env_args):
    if env_name != "cascade_reservoir":
        raise NotImplementedError(f"environment is not supported: {env_name}")
    from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv

    env = CascadeReservoirEnv(env_args)
    env.seed(seed)
    return env


def make_train_env(env_name, seed, n_threads, env_args):
    """Build the single environment required by sequential hydraulic sampling."""
    if n_threads != 1:
        raise ValueError("cascade_reservoir requires n_rollout_threads = 1 for prepare_step()")
    return ShareDummyVecEnv([lambda: _make_cascade_env(env_name, seed, env_args)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Build evaluation environment using the original evaluation seed rule."""
    if n_threads != 1:
        raise ValueError("cascade_reservoir requires n_eval_rollout_threads = 1 for prepare_step()")
    return ShareDummyVecEnv([lambda: _make_cascade_env(env_name, seed * 50000, env_args)])


def make_render_env(env_name, seed, env_args):
    """Reservoir render prints numerical states without an artificial delay."""
    env = _make_cascade_env(env_name, seed * 60000, env_args)
    return env, True, True, False, 1


def set_seed(args):
    """Seed the program."""

    if not args["seed_specify"]:
        args["seed"] = np.random.randint(
            1000,
            10000,
        )

    random.seed(
        args["seed"]
    )

    np.random.seed(
        args["seed"]
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        args["seed"]
    )

    torch.manual_seed(
        args["seed"]
    )

    torch.cuda.manual_seed(
        args["seed"]
    )

    torch.cuda.manual_seed_all(
        args["seed"]
    )


def get_num_agents(env, env_args, envs):
    """Get the number of reservoir agents."""
    if env != "cascade_reservoir":
        raise NotImplementedError(f"environment is not supported: {env}")
    return envs.n_agents
