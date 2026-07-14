from __future__ import annotations

import torch


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
