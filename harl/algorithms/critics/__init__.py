"""Reward critic registry; constraint critics are constructed separately."""
from harl.algorithms.critics.v_critic import VCritic

CRITIC_REGISTRY = {"happo": VCritic}
