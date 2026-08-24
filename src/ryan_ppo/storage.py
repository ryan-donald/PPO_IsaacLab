from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """
    single dataclass to store all the rollout buffers for the ppo update.
    """

    states: torch.Tensor  # (N, state_dim)
    actions: torch.Tensor  # (N, action_dim)
    log_probs_old: torch.Tensor  # (N,)
    returns: torch.Tensor  # (N,)
    advantages: torch.Tensor  # (N,) -- raw; update() normalizes over the batch
    values_old: torch.Tensor  # (N,)
    mus_old: torch.Tensor  # (N, action_dim)
    std_old: torch.Tensor  # (action_dim,) -- policy-level, not per-sample

    def __len__(self) -> int:
        return self.states.shape[0]


class RolloutStorage:
    """
    fixed-size buffers for one PPO rollout, shaped (num_steps, num_envs, ...).

    this class owns the rollout buffers. moves this code out of train.py,
    improving readability
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        state_dim: int,
        action_dim: int,
        num_terms: int,
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.states = torch.zeros(num_steps, num_envs, state_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.truncs = torch.zeros(num_steps, num_envs, device=device)
        self.term_rewards = torch.zeros(num_steps, num_envs, num_terms, device=device)
        self.mus = torch.zeros(num_steps, num_envs, action_dim, device=device)

    def add(
        self,
        step: int,
        *,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        trunc: torch.Tensor,
        term_reward: torch.Tensor,
        mu: torch.Tensor,
    ) -> None:
        self.states[step] = state
        self.actions[step] = action
        self.log_probs[step] = log_prob
        self.rewards[step] = reward
        self.dones[step] = done
        self.truncs[step] = trunc
        self.term_rewards[step] = term_reward
        self.mus[step] = mu

    def flatten(
        self,
        *,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor,
        std_old: torch.Tensor,
    ) -> RolloutBatch:
        """
        helper function to flatten the entire batch together, reducing repetition
        in the train.py file.
        """
        return RolloutBatch(
            states=self.states.reshape(-1, self.states.shape[-1]),
            actions=self.actions.reshape(-1, self.actions.shape[-1]),
            log_probs_old=self.log_probs.reshape(-1),
            returns=returns.reshape(-1),
            advantages=advantages.reshape(-1),
            values_old=values_old.reshape(-1),
            mus_old=self.mus.reshape(-1, self.mus.shape[-1]),
            std_old=std_old,
        )
