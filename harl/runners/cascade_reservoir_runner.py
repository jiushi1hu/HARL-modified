from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from harl.algorithms.critics.constraint_critic import (
    ConstraintCriticSet,
)
from harl.algorithms.lagrangian_manager import (
    LagrangianManager,
)
from harl.common.buffers.constraint_buffer import (
    ConstraintBuffer,
)
from harl.envs.cascade_reservoir.sequential_sampler import (
    SequentialSampler,
    SequentialSamplingResult,
)
from harl.envs.cascade_reservoir.s3_relaxation import (
    SafetyRecoveryResult,
)
from harl.runners.on_policy_ha_runner import (
    OnPolicyHARunner,
)
from harl.utils.trans_tools import _t2n
from harl.utils.configs_tools import save_config


def summarize_lagrangian_advantages(reward, constraints, multipliers, constraint_ids):
    """Read-only pre-update scales; these are advantages, not policy gradients."""
    reward = np.asarray(reward, dtype=np.float64)
    weighted = np.asarray(constraints, dtype=np.float64) * np.asarray(multipliers)
    penalty = weighted.sum(axis=-1, keepdims=True)
    combined = reward - penalty

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    info = {"advantage/reward_rms": rms(reward),
            "advantage/penalty_rms": rms(penalty),
            "advantage/combined_rms": rms(combined),
            "advantage/reward_std": float(np.std(reward)),
            "advantage/penalty_std": float(np.std(penalty)),
            "advantage/sign_flip_fraction": float(np.mean(reward * combined < 0))}
    # Undefined ratios are omitted rather than manufactured with an epsilon.
    if info["advantage/reward_rms"] > 0:
        info["advantage/penalty_to_reward_rms"] = rms(penalty) / rms(reward)
    for j, cid in enumerate(constraint_ids):
        info[f"advantage/constraint_rms/{cid}"] = rms(constraints[..., j])
        info[f"advantage/weighted_constraint_rms/{cid}"] = rms(weighted[..., j])
        info[f"advantage/lambda_snapshot/{cid}"] = float(multipliers[j])
    return info


_RESERVOIR_ORDER = (
    "WDD",
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_DOWNSTREAM_RESERVOIRS = (
    "BHT",
    "XLD",
    "XJB",
    "THR",
)

_STATE_ATOL = 1e-6
_LOG_PROB_ATOL = 1e-7
_OBSERVATION_ENCODING = "total_inflow_divided_by_map_max_release_v1"
_OBSERVATION_METADATA_FILE = "actor_observation_encoding.json"


class CascadeReservoirRunner(OnPolicyHARunner):

    def __init__(
        self,
        args,
        algo_args,
        env_args,
    ):
        self._constraint_components_ready = False
        self._deferred_constraint_restore = False

        self._validate_configuration(
            args=args,
            algo_args=algo_args,
            env_args=env_args,
        )

        super().__init__(
            args=args,
            algo_args=algo_args,
            env_args=env_args,
        )

        if self.num_agents != len(
            _RESERVOIR_ORDER
        ):
            raise ValueError(
                "cascade_reservoir requires "
                "exactly five agents"
            )

        self.raw_train_env = (
            self._unwrap_single_env(
                self.envs
            )
        )

        action_mappers = (
            self._get_action_mappers(
                self.raw_train_env
            )
        )

        self._reference_action_mappers = {
            reservoir_id:
            action_mappers[
                reservoir_id
            ]
            for reservoir_id
            in _RESERVOIR_ORDER
        }
        
        actors = {
            reservoir_id:
            self.actor[agent_id]
            for agent_id, reservoir_id
            in enumerate(
                _RESERVOIR_ORDER
            )
        }

        self.sequential_sampler = (
            SequentialSampler(
                actors=actors,
                action_mappers=(
                    action_mappers
                ),
            )
        )

        self._pending_sampling_result = None
        self._pending_constraint_values = None
        self._pending_constraint_rnn_states = None
        self._pending_step = None

        self.last_lagrangian_update = None

        if not self.algo_args[
            "render"
        ][
            "use_render"
        ]:
            self._init_constraint_components(
                self.raw_train_env
            )

            if (
                self._deferred_constraint_restore
                and self.algo_args[
                    "train"
                ][
                    "model_dir"
                ]
                is not None
            ):
                self._restore_constraint_components(
                    self.algo_args[
                        "train"
                    ][
                        "model_dir"
                    ]
                )

                self._deferred_constraint_restore = (
                    False
                )

    def _init_constraint_components(
        self,
        raw_env,
    ) -> None:

        if not hasattr(
            raw_env,
            "constraint_cost_evaluator",
        ):
            raise AttributeError(
                "CascadeReservoirEnv must define "
                "constraint_cost_evaluator"
            )

        constraint_evaluator = (
            raw_env.constraint_cost_evaluator
        )

        constraint_ids = tuple(
            constraint_evaluator
            .constraint_ids
        )

        budgets = np.asarray(
            constraint_evaluator.budgets,
            dtype=np.float32,
        )

        if len(
            constraint_ids
        ) == 0:
            raise ValueError(
                "at least one long-term "
                "constraint is required"
            )

        share_obs_space = (
            self.envs
            .share_observation_space[
                0
            ]
        )

        constraint_buffer_args = {
            **self.algo_args["train"],
            **self.algo_args["model"],
            **self.algo_args["algo"],
        }

        constraint_critic_args = {
            **self.algo_args["model"],
            **self.algo_args["algo"],
        }

        self.constraint_buffer = (
            ConstraintBuffer(
                args=(
                    constraint_buffer_args
                ),
                share_obs_space=(
                    share_obs_space
                ),
                constraint_ids=(
                    constraint_ids
                ),
                budgets=budgets,
                normalizers=[spec.normalizer for spec in constraint_evaluator.specs],
                tolerances=[spec.tolerance for spec in constraint_evaluator.specs],
            )
        )

        self.constraint_critic = (
            ConstraintCriticSet(
                critic_args=(
                    constraint_critic_args
                ),
                share_obs_space=(
                    share_obs_space
                ),
                constraint_ids=(
                    constraint_ids
                ),
                device=self.device,
                use_valuenorm=(
                    self.algo_args[
                        "train"
                    ][
                        "use_valuenorm"
                    ]
                ),
            )
        )

        self.lagrangian_manager = (
            LagrangianManager.from_config(
                env_args=self.env_args,
                constraint_ids=(
                    constraint_ids
                ),
                budgets=budgets,
            )
        )

        self.constraint_ids = (
            constraint_ids
        )

        self.num_constraints = len(
            constraint_ids
        )

        self._constraint_components_ready = (
            True
        )

    def run(self):
        if self.algo_args[
            "render"
        ][
            "use_render"
        ]:
            self.render()
            return

        print("start running")

        self.warmup()

        episodes = (
            int(
                self.algo_args[
                    "train"
                ][
                    "num_env_steps"
                ]
            )
            // self.algo_args[
                "train"
            ][
                "episode_length"
            ]
            // self.algo_args[
                "train"
            ][
                "n_rollout_threads"
            ]
        )

        self.logger.init(
            episodes
        )

        # Preserve the policy used by the first rollout for fair evaluation.
        self._save_training_checkpoint(0, initial=True)
        save_interval = self.algo_args["train"].get(
            "save_interval", self.algo_args["train"]["eval_interval"]
        )

        for episode in range(
            1,
            episodes + 1,
        ):
            if self.algo_args[
                "train"
            ][
                "use_linear_lr_decay"
            ]:
                for agent_id in range(
                    self.num_agents
                ):
                    self.actor[
                        agent_id
                    ].lr_decay(
                        episode,
                        episodes,
                    )

                self.critic.lr_decay(
                    episode,
                    episodes,
                )

                self.constraint_critic.lr_decay(
                    episode,
                    episodes,
                )

            self.logger.episode_init(
                episode
            )

            self.prep_rollout()

            for step in range(
                self.algo_args[
                    "train"
                ][
                    "episode_length"
                ]
            ):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(
                    step
                )

                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(
                    actions
                )

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.logger.per_step(
                    data
                )

                self.insert(
                    data
                )

            self.compute()

            self.prep_training()

            (
                actor_train_infos,
                critic_train_info,
            ) = self.train()

            # Save before evaluation so an evaluation failure cannot lose the
            # just-completed training update. The last update always persists.
            if episode % save_interval == 0 or episode == episodes:
                self._save_training_checkpoint(episode, final=episode == episodes)

            if (
                episode
                % self.algo_args[
                    "train"
                ][
                    "log_interval"
                ]
                == 0
            ):
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            if (
                episode
                % self.algo_args[
                    "train"
                ][
                    "eval_interval"
                ]
                == 0
            ):
                if self.algo_args[
                    "eval"
                ][
                    "use_eval"
                ]:
                    self.prep_rollout()
                    self.eval()

            self.after_update()

    def warmup(self):
        (
            _,
            share_obs,
            _,
        ) = self.envs.reset()

        share_obs = (
            self._validate_share_obs(
                share_obs
            )
        )

        self.critic_buffer.share_obs[
            0
        ] = share_obs[
            :,
            0,
        ].copy()

        self.constraint_buffer.initialize(
            share_obs[
                :,
                0,
            ]
        )

        for actor_buffer in (
            self.actor_buffer
        ):
            actor_buffer.rnn_states[
                0
            ].fill(
                0.0
            )

            actor_buffer.masks[
                0
            ].fill(
                1.0
            )

            actor_buffer.active_masks[
                0
            ].fill(
                1.0
            )

        self._clear_pending()

    @torch.no_grad()
    def collect(
        self,
        step,
    ):
        if (
            not self
            ._constraint_components_ready
        ):
            raise RuntimeError(
                "constraint components "
                "are not initialized"
            )

        if (
            self._pending_sampling_result
            is not None
        ):
            raise RuntimeError(
                "previous sampling result "
                "has not been inserted"
            )

        self._validate_buffer_step(
            step
        )

        # Build one read-only pre-action context.  No Actor has
        # sampled an action at this point.
        context = (
            self._get_sampling_context(
                self.raw_train_env
            )
        )

        # The environment state used by S2/S3, the reward critic
        # and every constraint critic must be the same x_t.
        current_share_obs = (
            self._validate_current_global_state(
                context=context,
                step=step,
            )
        )

        (
            value,
            rnn_state_critic,
        ) = self.critic.get_values(
            current_share_obs,
            self.critic_buffer
            .rnn_states_critic[
                step
            ],
            self.critic_buffer
            .masks[
                step
            ],
        )

        values = _t2n(
            value
        )

        rnn_states_critic = _t2n(
            rnn_state_critic
        )

        (
            constraint_rnn_states,
            constraint_masks,
        ) = (
            self._get_constraint_rollout_state(
                step
            )
        )

        if not np.array_equal(
            constraint_masks,
            self.critic_buffer
            .masks[
                step
            ],
        ):
            raise RuntimeError(
                "reward and constraint "
                "critic masks are out of sync"
            )

        (
            constraint_values,
            next_constraint_rnn_states,
        ) = (
            self.constraint_critic
            .get_values(
                current_share_obs,
                constraint_rnn_states,
                constraint_masks,
            )
        )

        constraint_values = _t2n(
            constraint_values
        )

        next_constraint_rnn_states = (
            _t2n(
                next_constraint_rnn_states
            )
        )

        actor_rnn_states = {
            reservoir_id:
            self.actor_buffer[
                agent_id
            ].rnn_states[
                step
            ].copy()
            for agent_id, reservoir_id
            in enumerate(
                _RESERVOIR_ORDER
            )
        }

        actor_rnn_masks = {
            reservoir_id:
            self.actor_buffer[
                agent_id
            ].masks[
                step
            ].copy()
            for agent_id, reservoir_id
            in enumerate(
                _RESERVOIR_ORDER
            )
        }

        # S4 actual sampling is hydraulic-order sequential:
        # WDD -> BHT -> XLD -> XJB -> THR.
        #
        # SequentialSampler currently accepts a batch/sequence of
        # SafetyRecoveryResult objects.  With one rollout thread,
        # the batch therefore contains exactly one result.
        sampling_result = (
            self.sequential_sampler.sample(
                base_observations=(
                    context[
                        "base_observations"
                    ]
                ),
                external_inflow_m3s=(
                    context[
                        "external_inflow_m3s"
                    ]
                ),
                interval_inflows_m3s=(
                    context[
                        "interval_inflows_m3s"
                    ]
                ),
                safety_results=(
                    context[
                        "safety_result"
                    ],
                ),
                rnn_states=(
                    actor_rnn_states
                ),
                rnn_masks=(
                    actor_rnn_masks
                ),
                deterministic=False,
            )
        )

        if not isinstance(
            sampling_result,
            SequentialSamplingResult,
        ):
            raise TypeError(
                "SequentialSampler must return "
                "SequentialSamplingResult"
            )

        next_actor_rnn_states = (
            np.stack(
                sampling_result
                .rnn_states,
                axis=1,
            )
        )

        if not np.all(
            np.isfinite(
                next_actor_rnn_states
            )
        ):
            raise RuntimeError(
                "Actor RNN states became "
                "non-finite during S4"
            )

        self._pending_sampling_result = (
            sampling_result
        )

        self._pending_constraint_values = (
            constraint_values
        )

        self._pending_constraint_rnn_states = (
            next_constraint_rnn_states
        )

        self._pending_step = int(
            step
        )

        return (
            values,
            sampling_result.actions,
            sampling_result.action_log_probs,
            next_actor_rnn_states,
            rnn_states_critic,
        )

    def insert(
        self,
        data,
    ):
        (
            _,
            share_obs,
            rewards,
            dones,
            infos,
            _,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        sampling_result = (
            self._pending_sampling_result
        )

        if sampling_result is None:
            raise RuntimeError(
                "insert called without "
                "a sampling result"
            )

        if (
            self._pending_constraint_values
            is None
            or self._pending_constraint_rnn_states
            is None
        ):
            raise RuntimeError(
                "constraint critic rollout "
                "data is missing"
            )

        if not np.array_equal(
            actions,
            sampling_result.actions,
        ):
            raise RuntimeError(
                "actions do not match "
                "sequential sampling result"
            )

        if not np.allclose(
            action_log_probs,
            sampling_result
            .action_log_probs,
            rtol=0.0,
            atol=_LOG_PROB_ATOL,
        ):
            raise RuntimeError(
                "action log probabilities "
                "do not match sequential "
                "sampling result"
            )

        share_obs = np.asarray(
            share_obs,
            dtype=np.float32,
        )

        rewards = np.asarray(
            rewards,
            dtype=np.float32,
        )

        dones = np.asarray(
            dones,
            dtype=bool,
        )

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        rnn_states = np.asarray(
            rnn_states,
            dtype=np.float32,
        ).copy()

        rnn_states_critic = np.asarray(
            rnn_states_critic,
            dtype=np.float32,
        ).copy()

        constraint_rnn_states = np.asarray(
            self._pending_constraint_rnn_states,
            dtype=np.float32,
        ).copy()

        constraint_values = np.asarray(
            self._pending_constraint_values,
            dtype=np.float32,
        )

        share_obs = (
            self._validate_share_obs(
                share_obs
            )
        )

        self._validate_synchronous_dones(
            dones
        )

        dones_env = np.all(
            dones,
            axis=1,
        )

        if np.any(
            dones_env
        ):
            rnn_states[
                dones_env
            ] = 0.0

            rnn_states_critic[
                dones_env
            ] = 0.0

            constraint_rnn_states[
                dones_env
            ] = 0.0

        masks = np.ones(
            (
                dones.shape[0],
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        masks[
            dones_env
        ] = 0.0

        active_masks = np.ones_like(
            masks
        )

        bad_masks = (
            self._build_bad_masks(
                infos
            )
        )

        (
            constraint_costs,
            constraint_active_flags,
            constraint_raw_violations,
        ) = (
            self._extract_constraint_feedback(
                infos
            )
        )

        for agent_id in range(
            self.num_agents
        ):
            actor_buffer = (
                self.actor_buffer[
                    agent_id
                ]
            )

            buffer_step = (
                actor_buffer.step
            )

            if (
                buffer_step
                != self._pending_step
            ):
                raise RuntimeError(
                    f"{_RESERVOIR_ORDER[agent_id]} "
                    "actor buffer is out of sync"
                )

            behavior_obs = (
                sampling_result
                .decision_observations[
                    agent_id
                ]
            )

            behavior_mask = (
                sampling_result
                .action_masks[
                    agent_id
                ]
            )

            if (
                behavior_obs.shape
                != actor_buffer.obs[
                    buffer_step
                ].shape
            ):
                raise ValueError(
                    f"{_RESERVOIR_ORDER[agent_id]}: "
                    "decision observation shape "
                    "does not match actor buffer"
                )

            if (
                actor_buffer
                .available_actions
                is None
            ):
                raise RuntimeError(
                    "cascade reservoir requires "
                    "Discrete actor actions"
                )

            if (
                behavior_mask.shape
                != actor_buffer
                .available_actions[
                    buffer_step
                ].shape
            ):
                raise ValueError(
                    f"{_RESERVOIR_ORDER[agent_id]}: "
                    "action mask shape does not "
                    "match actor buffer"
                )

            actor_buffer.obs[
                buffer_step
            ] = behavior_obs.copy()

            actor_buffer.available_actions[
                buffer_step
            ] = behavior_mask.copy()

            actor_buffer.actions[
                buffer_step
            ] = (
                sampling_result
                .actions[
                    :,
                    agent_id,
                    :,
                ].copy()
            )

            actor_buffer.action_log_probs[
                buffer_step
            ] = (
                sampling_result
                .action_log_probs[
                    :,
                    agent_id,
                    :,
                ].copy()
            )

            actor_buffer.rnn_states[
                buffer_step + 1
            ] = rnn_states[
                :,
                agent_id,
            ].copy()

            actor_buffer.masks[
                buffer_step + 1
            ] = masks[
                :,
                agent_id,
            ].copy()

            actor_buffer.active_masks[
                buffer_step + 1
            ] = active_masks[
                :,
                agent_id,
            ].copy()

            actor_buffer.step = (
                buffer_step + 1
            ) % actor_buffer.episode_length

        self.critic_buffer.insert(
            share_obs[
                :,
                0,
            ],
            rnn_states_critic,
            values,
            rewards[
                :,
                0,
            ],
            masks[
                :,
                0,
            ],
            bad_masks,
        )

        self.constraint_buffer.insert(
            share_obs=(
                share_obs[
                    :,
                    0,
                ]
            ),
            rnn_states_critic=(
                constraint_rnn_states
            ),
            value_preds=(
                constraint_values
            ),
            costs=(
                constraint_costs
            ),
            raw_violations=constraint_raw_violations,
            active_flags=(
                constraint_active_flags
            ),
            masks=(
                masks[
                    :,
                    0,
                ]
            ),
            bad_masks=(
                bad_masks
            ),
        )

        self._clear_pending()

    @torch.no_grad()
    def compute(self):
        super().compute()

        self.constraint_critic.compute_returns(
            self.constraint_buffer
        )

    def train(self):
        # Logger indexes this list by reservoir/agent ID, independently
        # of the randomized HAPPO parameter update order.
        actor_train_infos = [None] * self.num_agents

        factor = np.ones(
            (
                self.algo_args[
                    "train"
                ][
                    "episode_length"
                ],
                self.algo_args[
                    "train"
                ][
                    "n_rollout_threads"
                ],
                1,
            ),
            dtype=np.float32,
        )

        if self.value_normalizer is not None:
            reward_advantages = (
                self.critic_buffer
                .returns[
                    :-1
                ]
                - self.value_normalizer
                .denormalize(
                    self.critic_buffer
                    .value_preds[
                        :-1
                    ]
                )
            )
        else:
            reward_advantages = (
                self.critic_buffer
                .returns[
                    :-1
                ]
                - self.critic_buffer
                .value_preds[
                    :-1
                ]
            )

        reward_advantages = np.asarray(
            reward_advantages,
            dtype=np.float32,
        )

        constraint_advantages = (
            self.constraint_critic
            .compute_advantages(
                self.constraint_buffer
            )
        )

        lambda_snapshot = (
            self.lagrangian_manager
            .snapshot()
        )

        lagrangian_advantages = (
            self.lagrangian_manager
            .combine_advantages(
                reward_advantages=(
                    reward_advantages
                ),
                constraint_advantages=(
                    constraint_advantages
                ),
                multipliers=(
                    lambda_snapshot
                ),
            )
        )

        # Freeze A_L using rollout values and pre-update normalizers above.
        # Critics may update their normalizers; never recompute this batch's
        # Actor advantages after these updates.
        advantage_diagnostics = summarize_lagrangian_advantages(
            reward_advantages, constraint_advantages, lambda_snapshot, self.constraint_ids)
        reward_critic_train_info = (
            self.critic.train(
                self.critic_buffer,
                self.value_normalizer,
            )
        )

        constraint_critic_train_info = (
            self.constraint_critic.train(
                self.constraint_buffer
            )
        )

        if self.fixed_order:
            agent_order = list(
                range(
                    self.num_agents
                )
            )
        else:
            agent_order = list(
                torch.randperm(
                    self.num_agents
                ).numpy()
            )

        for agent_id in (
            agent_order
        ):
            actor_buffer = (
                self.actor_buffer[
                    agent_id
                ]
            )

            actor_buffer.update_factor(
                factor
            )

            if (
                actor_buffer
                .available_actions
                is None
            ):
                available_actions = None
            else:
                available_actions = (
                    actor_buffer
                    .available_actions[
                        :-1
                    ]
                    .reshape(
                        -1,
                        *actor_buffer
                        .available_actions
                        .shape[
                            2:
                        ],
                    )
                )

            actor_train_info = (
                self.actor[
                    agent_id
                ].train(
                    actor_buffer,
                    lagrangian_advantages
                    .copy(),
                    "EP",
                )
            )

            (
                new_actions_logprob,
                _,
                _,
            ) = (
                self.actor[
                    agent_id
                ].evaluate_actions(
                    actor_buffer
                    .obs[
                        :-1
                    ]
                    .reshape(
                        -1,
                        *actor_buffer
                        .obs.shape[
                            2:
                        ],
                    ),
                    actor_buffer
                    .rnn_states[
                        0:1
                    ]
                    .reshape(
                        -1,
                        *actor_buffer
                        .rnn_states
                        .shape[
                            2:
                        ],
                    ),
                    actor_buffer
                    .actions
                    .reshape(
                        -1,
                        *actor_buffer
                        .actions
                        .shape[
                            2:
                        ],
                    ),
                    actor_buffer
                    .masks[
                        :-1
                    ]
                    .reshape(
                        -1,
                        *actor_buffer
                        .masks.shape[
                            2:
                        ],
                    ),
                    available_actions,
                    actor_buffer
                    .active_masks[
                        :-1
                    ]
                    .reshape(
                        -1,
                        *actor_buffer
                        .active_masks
                        .shape[
                            2:
                        ],
                    ),
                )
            )

            # Denominator is the actual stored behavior policy probability,
            # matching HAPPO.update(), not a fresh pre-update policy evaluation.
            old_actions_logprob = torch.as_tensor(
                actor_buffer.action_log_probs.reshape(
                    -1, *actor_buffer.action_log_probs.shape[2:]
                ),
                dtype=new_actions_logprob.dtype,
                device=new_actions_logprob.device,
            )

            factor = (
                factor
                * _t2n(
                    getattr(
                        torch,
                        self.action_aggregation,
                    )(
                        torch.exp(
                            new_actions_logprob
                            - old_actions_logprob
                        ),
                        dim=-1,
                    )
                    .reshape(
                        self.algo_args[
                            "train"
                        ][
                            "episode_length"
                        ],
                        self.algo_args[
                            "train"
                        ][
                            "n_rollout_threads"
                        ],
                        1,
                    )
                )
            )

            actor_train_infos[agent_id] = actor_train_info

        constraint_statistics = (
            self.constraint_buffer
            .compute_discounted_active_statistics()
        )

        lagrangian_update = (
            self.lagrangian_manager
            .update(
                constraint_statistics
            )
        )

        self.last_lagrangian_update = (
            lagrangian_update
        )

        critic_train_info = dict(
            reward_critic_train_info
        )
        critic_train_info.update(advantage_diagnostics)

        for key, value in (
            constraint_critic_train_info
            .items()
        ):
            critic_train_info[
                f"constraint_critic/{key}"
            ] = value

        for constraint_index, (
            constraint_id
        ) in enumerate(
            self.constraint_ids
        ):
            critic_train_info[f"lambda_before/{constraint_id}"] = float(
                lagrangian_update.old_multipliers[constraint_index]
            )
            critic_train_info[
                f"lambda/{constraint_id}"
            ] = float(
                lagrangian_update
                .new_multipliers[
                    constraint_index
                ]
            )

            critic_train_info[
                f"budget/{constraint_id}"
            ] = float(
                lagrangian_update
                .budgets[
                    constraint_index
                ]
            )

            critic_train_info[
                f"active/{constraint_id}"
            ] = float(
                lagrangian_update
                .valid_flags[
                    constraint_index
                ]
            )

            if lagrangian_update.valid_flags[
                constraint_index
            ]:
                critic_train_info[
                    f"mean_cost/{constraint_id}"
                ] = float(
                    lagrangian_update
                    .mean_costs[
                        constraint_index
                    ]
                )

                critic_train_info[
                    f"budget_gap/{constraint_id}"
                ] = float(
                    lagrangian_update
                    .budget_gaps[
                        constraint_index
                    ]
                )

        return (
            actor_train_infos,
            critic_train_info,
        )

    def after_update(self):
        super().after_update()

        self.constraint_buffer.after_update()

        self._clear_pending()

    def prep_rollout(self):
        super().prep_rollout()

        if self._constraint_components_ready:
            self.constraint_critic.prep_rollout()

    def prep_training(self):
        super().prep_training()

        if self._constraint_components_ready:
            self.constraint_critic.prep_training()

    def _save_training_checkpoint(self, completed_updates, *, initial=False, final=False):
        """Save evaluation weights and their provenance; not an exact resume state."""
        original_dir = self.save_dir
        target = Path(original_dir) / "initial" if initial else Path(original_dir)
        if initial:
            # Never silently replace the baseline if run() is called twice.
            target.mkdir(parents=True, exist_ok=False)
        try:
            self.save_dir = target
            metadata_path = target / "checkpoint_metadata.json"
            # An interrupted replacement must not retain an old completion marker.
            metadata_path.unlink(missing_ok=True)
            self.save()
            save_config(self.args, self.algo_args, self.env_args, target)
            metadata = {
                "kind": "run_start" if initial else ("final" if final else "periodic"),
                "completed_updates_in_run": completed_updates,
                "training_steps_in_run": completed_updates
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"],
                "source_model_dir": self.algo_args["train"]["model_dir"],
                "observation_encoding": _OBSERVATION_ENCODING,
                "exact_training_resume": False,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        finally:
            self.save_dir = original_dir
        print(f"Saved {metadata['kind']} checkpoint: {target}", flush=True)

    def save(self):
        super().save()

        (Path(self.save_dir) / _OBSERVATION_METADATA_FILE).write_text(
            json.dumps({"encoding": _OBSERVATION_ENCODING}) + "\n",
            encoding="utf-8",
        )

        if (
            not self.algo_args[
                "render"
            ][
                "use_render"
            ]
            and self._constraint_components_ready
        ):
            self.constraint_critic.save(
                self.save_dir
            )

            self.lagrangian_manager.save(
                self.save_dir
            )

    def restore(self):
        self._validate_observation_checkpoint(
            self.algo_args["train"]["model_dir"]
        )
        super().restore()

        if self.algo_args[
            "render"
        ][
            "use_render"
        ]:
            return

        if getattr(
            self,
            "_constraint_components_ready",
            False,
        ):
            self._restore_constraint_components(
                self.algo_args[
                    "train"
                ][
                    "model_dir"
                ]
            )
        else:
            self._deferred_constraint_restore = (
                True
            )

    @staticmethod
    def _validate_observation_checkpoint(model_dir):
        metadata_path = Path(model_dir) / _OBSERVATION_METADATA_FILE
        if not metadata_path.is_file():
            raise ValueError(
                "Checkpoint has no actor observation encoding metadata. "
                "Legacy cascade models used raw downstream inflow; they "
                "cannot be resumed with normalized inflow. Start a new "
                "experiment with model_dir=None; keep the old checkpoint "
                "for evaluation with its original code."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("encoding") != _OBSERVATION_ENCODING:
            raise ValueError("Checkpoint actor observation encoding is incompatible")

    def _restore_constraint_components(
        self,
        model_dir,
    ) -> None:

        self.constraint_critic.restore(
            model_dir
        )

        self.lagrangian_manager.restore(
            model_dir
        )

    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()

        raw_eval_env = (
            self._unwrap_single_env(
                self.eval_envs
            )
        )

        self._validate_environment_action_mappers(
            raw_eval_env
        )

        self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                1,
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        eval_masks = np.ones(
            (
                1,
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        eval_episode = 0

        while (
            eval_episode
            < self.algo_args[
                "eval"
            ][
                "eval_episodes"
            ]
        ):
            context = (
                self._get_sampling_context(
                    raw_eval_env
                )
            )

            sampling_result = (
                self._sample_context(
                    context=context,
                    rnn_states=(
                        eval_rnn_states
                    ),
                    rnn_masks=(
                        eval_masks
                    ),
                    deterministic=True,
                )
            )

            eval_rnn_states = (
                np.stack(
                    sampling_result
                    .rnn_states,
                    axis=1,
                )
            )

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(
                sampling_result.actions
            )

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )

            self.logger.eval_per_step(
                eval_data
            )

            eval_dones = np.asarray(
                eval_dones,
                dtype=bool,
            )

            self._validate_synchronous_dones(
                eval_dones
            )

            done = bool(
                np.all(
                    eval_dones[
                        0
                    ]
                )
            )

            eval_masks.fill(
                1.0
            )

            if done:
                eval_rnn_states.fill(
                    0.0
                )

                eval_masks.fill(
                    0.0
                )

                eval_episode += 1

                self.logger.eval_thread_done(
                    0
                )

        self.logger.eval_log(
            eval_episode
        )

    @torch.no_grad()
    def render(self):
        raw_env = (
            self.raw_train_env
        )

        self._validate_environment_action_mappers(
            raw_env
        )

        print(
            "start rendering"
        )

        for _ in range(
            self.algo_args[
                "render"
            ][
                "render_episodes"
            ]
        ):
            raw_env.reset()

            rnn_states = np.zeros(
                (
                    1,
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            rnn_masks = np.ones(
                (
                    1,
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )

            total_reward = 0.0

            while True:
                context = (
                    self._get_sampling_context(
                        raw_env
                    )
                )

                sampling_result = (
                    self._sample_context(
                        context=context,
                        rnn_states=(
                            rnn_states
                        ),
                        rnn_masks=(
                            rnn_masks
                        ),
                        deterministic=True,
                    )
                )

                rnn_states = np.stack(
                    sampling_result
                    .rnn_states,
                    axis=1,
                )

                (
                    _,
                    _,
                    rewards,
                    dones,
                    _,
                    _,
                ) = raw_env.step(
                    sampling_result.actions[
                        0
                    ]
                )

                rewards = np.asarray(
                    rewards,
                    dtype=np.float64,
                )

                total_reward += float(
                    rewards[
                        0,
                        0,
                    ]
                )

                if self.manual_render:
                    raw_env.render()

                if self.manual_delay:
                    time.sleep(
                        0.1
                    )

                done = bool(
                    np.all(
                        np.asarray(
                            dones,
                            dtype=bool,
                        )
                    )
                )

                if done:
                    break

            print(
                "total reward of this episode: "
                f"{total_reward}"
            )

    def _sample_context(
        self,
        context,
        rnn_states,
        rnn_masks,
        deterministic,
    ):
        rnn_states = np.asarray(
            rnn_states,
            dtype=np.float32,
        )

        rnn_masks = np.asarray(
            rnn_masks,
            dtype=np.float32,
        )

        expected_rnn_shape = (
            1,
            self.num_agents,
            self.recurrent_n,
            self.rnn_hidden_size,
        )

        if (
            rnn_states.shape
            != expected_rnn_shape
        ):
            raise ValueError(
                "Actor RNN states have "
                "invalid shape; expected "
                f"{expected_rnn_shape}, "
                f"received {rnn_states.shape}"
            )

        expected_mask_shape = (
            1,
            self.num_agents,
            1,
        )

        if (
            rnn_masks.shape
            != expected_mask_shape
        ):
            raise ValueError(
                "Actor RNN masks have "
                "invalid shape; expected "
                f"{expected_mask_shape}, "
                f"received {rnn_masks.shape}"
            )

        if not np.all(
            np.isfinite(
                rnn_states
            )
        ):
            raise ValueError(
                "Actor RNN states contain "
                "non-finite values"
            )

        if not np.all(
            (
                rnn_masks == 0.0
            )
            | (
                rnn_masks == 1.0
            )
        ):
            raise ValueError(
                "Actor RNN masks "
                "must be binary"
            )

        state_mapping = {
            reservoir_id:
            rnn_states[
                :,
                agent_id,
            ]
            for agent_id, reservoir_id
            in enumerate(
                _RESERVOIR_ORDER
            )
        }

        mask_mapping = {
            reservoir_id:
            rnn_masks[
                :,
                agent_id,
            ]
            for agent_id, reservoir_id
            in enumerate(
                _RESERVOIR_ORDER
            )
        }

        result = (
            self.sequential_sampler
            .sample(
                base_observations=(
                    context[
                        "base_observations"
                    ]
                ),
                external_inflow_m3s=(
                    context[
                        "external_inflow_m3s"
                    ]
                ),
                interval_inflows_m3s=(
                    context[
                        "interval_inflows_m3s"
                    ]
                ),
                safety_results=(
                    context[
                        "safety_result"
                    ],
                ),
                rnn_states=(
                    state_mapping
                ),
                rnn_masks=(
                    mask_mapping
                ),
                deterministic=(
                    deterministic
                ),
            )
        )

        if not isinstance(
            result,
            SequentialSamplingResult,
        ):
            raise TypeError(
                "SequentialSampler must "
                "return "
                "SequentialSamplingResult"
            )

        return result

    def _get_constraint_rollout_state(
        self,
        step,
    ):
        rnn_states = np.stack(
            [
                self.constraint_buffer
                .get_buffer(
                    constraint_index
                )
                .rnn_states_critic[
                    step
                ]
                for constraint_index
                in range(
                    self.num_constraints
                )
            ],
            axis=1,
        )

        masks = (
            self.constraint_buffer
            .get_buffer(
                0
            )
            .masks[
                step
            ]
            .copy()
        )

        for constraint_index in range(
            1,
            self.num_constraints,
        ):
            other_mask = (
                self.constraint_buffer
                .get_buffer(
                    constraint_index
                )
                .masks[
                    step
                ]
            )

            if not np.array_equal(
                masks,
                other_mask,
            ):
                raise RuntimeError(
                    "constraint critic masks "
                    "are out of sync"
                )

        return (
            rnn_states,
            masks,
        )

    def _extract_constraint_feedback(
        self,
        infos,
    ):
        num_threads = len(
            infos
        )

        fields = ("constraint_costs", "constraint_active_flags", "constraint_raw_violations")
        matrices = [np.empty((num_threads, self.num_constraints), dtype=np.float32) for _ in fields]
        for thread_id, agent_infos in enumerate(infos):
            if len(agent_infos) != self.num_agents:
                raise ValueError("infos must contain one dictionary per reservoir")
            reference = None
            for info in agent_infos:
                if not isinstance(info, Mapping):
                    raise TypeError("environment info must be a mapping")
                if tuple(info["constraint_ids"]) != self.constraint_ids:
                    raise ValueError("environment constraint IDs do not match runner")
                vectors = [np.asarray(info[key], dtype=np.float32) for key in fields]
                for key, vector in zip(fields, vectors):
                    if vector.shape != (self.num_constraints,) or not np.isfinite(vector).all():
                        raise ValueError(key + " has invalid shape or non-finite values")
                costs, flags, raw = vectors
                expected = self.constraint_buffer.prepare_training_costs(raw[None], flags[None])[0]
                if (np.any(costs < 0.0) or np.any(costs[flags == 0.0] != 0.0)
                        or not np.allclose(costs, expected, rtol=1e-6, atol=1e-7)):
                    raise ValueError("environment cost disagrees with saved raw violation/active flag")
                if reference is None:
                    reference = vectors
                elif any(not np.array_equal(a, b) for a, b in zip(reference, vectors)):
                    raise ValueError("constraint feedback differs between agents")
            for matrix, vector in zip(matrices, reference):
                matrix[thread_id] = vector
        return tuple(matrices)

    @staticmethod
    def _get_sampling_context(
        raw_env,
    ):
        if not hasattr(
            raw_env,
            "prepare_step",
        ):
            raise AttributeError(
                "CascadeReservoirEnv must "
                "implement prepare_step()"
            )

        context = (
            raw_env.prepare_step()
        )

        if not isinstance(
            context,
            Mapping,
        ):
            raise TypeError(
                "prepare_step() must "
                "return a mapping"
            )

        required_keys = {
            "base_observations",
            "share_observation",
            "external_inflow_m3s",
            "interval_inflows_m3s",
            "safety_result",
        }

        missing = (
            required_keys
            - set(
                context
            )
        )

        if missing:
            raise KeyError(
                "prepare_step() is missing "
                f"{missing}"
            )

        base_observations = (
            context[
                "base_observations"
            ]
        )

        if not isinstance(
            base_observations,
            Mapping,
        ):
            raise TypeError(
                "base_observations "
                "must be a mapping"
            )

        if (
            set(
                base_observations
            )
            != set(
                _RESERVOIR_ORDER
            )
        ):
            raise ValueError(
                "base_observations must "
                "contain WDD, BHT, XLD, "
                "XJB and THR"
            )

        prepared_base = []

        for reservoir_id in (
            _RESERVOIR_ORDER
        ):
            observation = np.asarray(
                base_observations[
                    reservoir_id
                ],
                dtype=np.float32,
            )

            if (
                observation.ndim != 2
                or observation.shape[
                    0
                ]
                != 1
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "base observation must "
                    "have shape "
                    "(1, obs_dim)"
                )

            if (
                observation.shape[
                    1
                ]
                <= 0
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "base observation "
                    "cannot be empty"
                )

            if not np.all(
                np.isfinite(
                    observation
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "base observation "
                    "contains non-finite values"
                )

            prepared_base.append(
                observation
            )

        share_observation = np.asarray(
            context[
                "share_observation"
            ],
            dtype=np.float32,
        )

        if (
            share_observation.ndim != 2
            or share_observation.shape[
                0
            ]
            != 1
            or share_observation.shape[
                1
            ]
            <= 0
        ):
            raise ValueError(
                "share_observation must "
                "have shape "
                "(1, state_dim)"
            )

        if not np.all(
            np.isfinite(
                share_observation
            )
        ):
            raise ValueError(
                "share_observation contains "
                "non-finite values"
            )

        reconstructed_state = (
            np.concatenate(
                prepared_base,
                axis=1,
            )
        )

        if (
            reconstructed_state.shape
            != share_observation.shape
        ):
            raise ValueError(
                "share_observation dimension "
                "does not match concatenated "
                "base observations"
            )

        if not np.allclose(
            reconstructed_state,
            share_observation,
            rtol=0.0,
            atol=_STATE_ATOL,
        ):
            raise RuntimeError(
                "share_observation is "
                "inconsistent with current "
                "base observations"
            )

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
            raise ValueError(
                "external_inflow_m3s "
                "must be finite and "
                "nonnegative"
            )

        interval_inflows = (
            context[
                "interval_inflows_m3s"
            ]
        )

        if not isinstance(
            interval_inflows,
            Mapping,
        ):
            raise TypeError(
                "interval_inflows_m3s "
                "must be a mapping"
            )

        if (
            set(
                interval_inflows
            )
            != set(
                _DOWNSTREAM_RESERVOIRS
            )
        ):
            raise ValueError(
                "interval_inflows_m3s must "
                "contain BHT, XLD, XJB "
                "and THR"
            )

        for reservoir_id in (
            _DOWNSTREAM_RESERVOIRS
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
                raise ValueError(
                    f"{reservoir_id}: "
                    "interval inflow must "
                    "be finite and "
                    "nonnegative"
                )

        if not isinstance(
            context[
                "safety_result"
            ],
            SafetyRecoveryResult,
        ):
            raise TypeError(
                "safety_result must be "
                "SafetyRecoveryResult"
            )

        return context

    def _validate_current_global_state(
        self,
        *,
        context,
        step,
    ):
        """Verify one authoritative pre-action centralized state x_t."""

        context_state = np.asarray(
            context[
                "share_observation"
            ],
            dtype=np.float32,
        )

        reward_state = np.asarray(
            self.critic_buffer
            .share_obs[
                step
            ],
            dtype=np.float32,
        )

        if (
            context_state.shape
            != reward_state.shape
        ):
            raise ValueError(
                "prepare_step global state "
                "shape does not match "
                "reward critic buffer"
            )

        if not np.allclose(
            context_state,
            reward_state,
            rtol=0.0,
            atol=_STATE_ATOL,
        ):
            raise RuntimeError(
                "prepare_step global state "
                "does not match reward critic "
                "buffer state at the current "
                "rollout step"
            )

        for constraint_index in range(
            self.num_constraints
        ):
            constraint_state = np.asarray(
                self.constraint_buffer
                .get_buffer(
                    constraint_index
                )
                .share_obs[
                    step
                ],
                dtype=np.float32,
            )

            if (
                constraint_state.shape
                != context_state.shape
            ):
                raise ValueError(
                    "constraint critic state "
                    "shape does not match "
                    "prepare_step global state"
                )

            if not np.allclose(
                constraint_state,
                context_state,
                rtol=0.0,
                atol=_STATE_ATOL,
            ):
                raise RuntimeError(
                    "constraint critic state "
                    "is out of sync with the "
                    "current environment state"
                )

        return context_state.copy()

    @staticmethod
    def _get_action_mappers(
        raw_env,
    ):
        if not hasattr(
            raw_env,
            "get_action_mappers",
        ):
            raise AttributeError(
                "CascadeReservoirEnv must "
                "implement "
                "get_action_mappers()"
            )

        action_mappers = (
            raw_env
            .get_action_mappers()
        )

        if not isinstance(
            action_mappers,
            Mapping,
        ):
            raise TypeError(
                "get_action_mappers() must "
                "return a mapping"
            )

        if set(
            action_mappers
        ) != set(
            _RESERVOIR_ORDER
        ):
            raise ValueError(
                "action mappers must contain "
                "WDD, BHT, XLD, XJB and THR"
            )

        return action_mappers

    def _validate_environment_action_mappers(
        self,
        raw_env,
    ) -> None:
        action_mappers = (
            self._get_action_mappers(
                raw_env
            )
        )

        for reservoir_id in (
            _RESERVOIR_ORDER
        ):
            reference = (
                self
                ._reference_action_mappers[
                    reservoir_id
                ]
            )

            candidate = (
                action_mappers[
                    reservoir_id
                ]
            )

            if (
                int(
                    candidate.num_actions
                )
                != int(
                    reference.num_actions
                )
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "environment action count "
                    "does not match "
                    "training sampler"
                )

            reference_grid = np.asarray(
                reference
                .candidate_releases_m3s,
                dtype=np.float64,
            )

            candidate_grid = np.asarray(
                candidate
                .candidate_releases_m3s,
                dtype=np.float64,
            )

            if not np.array_equal(
                candidate_grid,
                reference_grid,
            ):
                raise ValueError(
                    f"{reservoir_id}: "
                    "environment release grid "
                    "does not match "
                    "training sampler"
                )

    @staticmethod
    def _unwrap_single_env(
        env_container,
    ):
        if hasattr(
            env_container,
            "envs",
        ):
            envs = (
                env_container.envs
            )

            if len(
                envs
            ) != 1:
                raise ValueError(
                    "cascade reservoir runner "
                    "requires one environment "
                    "instance"
                )

            return envs[
                0
            ]

        if hasattr(
            env_container,
            "remotes",
        ):
            raise RuntimeError(
                "ShareSubprocVecEnv cannot "
                "directly expose prepare_step(); "
                "use n_rollout_threads = 1"
            )

        return env_container

    def _validate_buffer_step(
        self,
        step,
    ):
        for agent_id, actor_buffer in enumerate(
            self.actor_buffer
        ):
            if (
                actor_buffer.step
                != step
            ):
                raise RuntimeError(
                    f"{_RESERVOIR_ORDER[agent_id]} "
                    "actor buffer is out of sync"
                )

        if (
            self.critic_buffer.step
            != step
        ):
            raise RuntimeError(
                "reward critic buffer "
                "is out of sync"
            )

        if (
            self.constraint_buffer.step
            != step
        ):
            raise RuntimeError(
                "constraint buffer "
                "is out of sync"
            )

    def _validate_share_obs(
        self,
        share_obs,
    ):
        share_obs = np.asarray(
            share_obs,
            dtype=np.float32,
        )

        if share_obs.ndim != 3:
            raise ValueError(
                "share_obs must have shape "
                "(n_threads, n_agents, "
                "state_dim)"
            )

        if (
            share_obs.shape[
                0
            ]
            != 1
        ):
            raise ValueError(
                "cascade reservoir currently "
                "requires one rollout thread"
            )

        if (
            share_obs.shape[
                1
            ]
            != self.num_agents
        ):
            raise ValueError(
                "share_obs agent dimension "
                "must equal five"
            )

        if (
            share_obs.shape[
                2
            ]
            <= 0
        ):
            raise ValueError(
                "share_obs state dimension "
                "cannot be empty"
            )

        if not np.all(
            np.isfinite(
                share_obs
            )
        ):
            raise ValueError(
                "share_obs contains "
                "non-finite values"
            )

        reference = (
            share_obs[
                :,
                0:1,
                :
            ]
        )

        if not np.allclose(
            share_obs,
            reference,
            rtol=0.0,
            atol=_STATE_ATOL,
        ):
            raise ValueError(
                "EP share_obs must be "
                "identical across all "
                "five agents"
            )

        return share_obs

    def _validate_synchronous_dones(
        self,
        dones,
    ):
        dones = np.asarray(
            dones,
            dtype=bool,
        )

        if dones.ndim != 2:
            raise ValueError(
                "dones must have shape "
                "(n_threads, n_agents)"
            )

        if (
            dones.shape[
                1
            ]
            != self.num_agents
        ):
            raise ValueError(
                "invalid dones agent "
                "dimension"
            )

        reference = dones[
            :,
            0:1,
        ]

        if not np.all(
            dones == reference
        ):
            raise RuntimeError(
                "cascade reservoir agents "
                "must terminate "
                "synchronously"
            )

    @staticmethod
    def _build_bad_masks(
        infos,
    ):
        bad_masks = []

        for thread_info in infos:
            if (
                len(
                    thread_info
                )
                == 0
            ):
                raise ValueError(
                    "infos cannot contain "
                    "an empty agent list"
                )

            first_info = (
                thread_info[
                    0
                ]
            )

            if not isinstance(
                first_info,
                Mapping,
            ):
                raise TypeError(
                    "agent info must "
                    "be a mapping"
                )

            bad_transition = bool(
                first_info.get(
                    "bad_transition",
                    False,
                )
            )

            bad_masks.append(
                [
                    0.0
                    if bad_transition
                    else 1.0
                ]
            )

        return np.asarray(
            bad_masks,
            dtype=np.float32,
        )

    def _clear_pending(
        self,
    ):
        self._pending_sampling_result = None
        self._pending_constraint_values = None
        self._pending_constraint_rnn_states = None
        self._pending_step = None

    @staticmethod
    def _validate_configuration(
        args,
        algo_args,
        env_args,
    ):
        save_interval = algo_args["train"].get(
            "save_interval", algo_args["train"]["eval_interval"]
        )
        if isinstance(save_interval, bool) or not isinstance(save_interval, int) or save_interval <= 0:
            raise ValueError("save_interval must be a positive integer")

        if (
            args[
                "env"
            ]
            != "cascade_reservoir"
        ):
            raise ValueError(
                "CascadeReservoirRunner "
                "is only for "
                "cascade_reservoir"
            )

        if (
            args[
                "algo"
            ]
            != "happo"
        ):
            raise ValueError(
                "cascade reservoir runner "
                "currently requires HAPPO"
            )

        if algo_args[
            "algo"
        ][
            "share_param"
        ]:
            raise ValueError(
                "cascade reservoir requires "
                "independent actors"
            )

        if algo_args[
            "algo"
        ][
            "fixed_order"
        ]:
            raise ValueError(
                "HAPPO parameter update order "
                "must remain randomized"
            )

        if (
            env_args.get(
                "state_type",
                "EP",
            )
            != "EP"
        ):
            raise ValueError(
                "cascade reservoir requires "
                "EP centralized state"
            )

        if (
            not algo_args[
                "render"
            ][
                "use_render"
            ]
            and algo_args[
                "train"
            ][
                "n_rollout_threads"
            ]
            != 1
        ):
            raise ValueError(
                "current cascade implementation "
                "requires "
                "n_rollout_threads = 1"
            )

        if (
            not algo_args[
                "render"
            ][
                "use_render"
            ]
            and algo_args[
                "eval"
            ][
                "use_eval"
            ]
            and algo_args[
                "eval"
            ][
                "n_eval_rollout_threads"
            ]
            != 1
        ):
            raise ValueError(
                "current cascade evaluation "
                "requires "
                "n_eval_rollout_threads = 1"
            )
