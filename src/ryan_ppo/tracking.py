from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutStats:
    avg_reward: float
    min_reward: float
    max_reward: float
    std_reward: float
    avg_entropy: float
    num_episodes: int
    term_rewards: dict[str, float]


class EpisodeTracker:
    """accumulates per-env episode returns and per-term rewards across a rollout,
    then reduces over the episodes that completed.

    accumulators are used for statistics about completed episodes.

    when a rollout has no completed episodes the previous iteration's reward stats
    are forward-filled (so logs/TUI don't flicker to zero), but num_episodes is
    reported as zero so cumulative episode counts stay accurate.
    """

    def __init__(
        self,
        num_envs: int,
        term_names,
        device: torch.device,
    ) -> None:
        self.term_names = list(term_names)
        self.num_terms = len(self.term_names)

        # running per-env returns; persist across rollouts so episodes longer than
        # a single rollout accumulate correctly. zeroed per-env as each finishes.
        self.current_rewards = torch.zeros(num_envs, device=device)
        self.current_terms = torch.zeros(num_envs, self.num_terms, device=device)

        # per-rollout aggregates over completed episodes (reset each summarize()).
        self.ep_sum = torch.zeros((), device=device)
        self.ep_sumsq = torch.zeros((), device=device)
        self.ep_count = torch.zeros((), device=device)
        self.term_sums = torch.zeros(self.num_terms, device=device)
        self.ep_min = torch.full((), float("inf"), device=device)
        self.ep_max = torch.full((), float("-inf"), device=device)

        # forward-filled on empty rollouts; mutated in place each summarize().
        self.last = RolloutStats(
            avg_reward=0.0,
            min_reward=0.0,
            max_reward=0.0,
            std_reward=0.0,
            avg_entropy=0.0,
            num_episodes=0,
            term_rewards={name: 0.0 for name in self.term_names},
        )

    def reset_accumulators(self) -> None:
        self.ep_sum.zero_()
        self.ep_sumsq.zero_()
        self.ep_count.zero_()
        self.term_sums.zero_()
        self.ep_min.fill_(float("inf"))
        self.ep_max.fill_(float("-inf"))

    def record_step(
        self,
        reward: torch.Tensor,
        done: torch.Tensor,
        step_term_rewards: torch.Tensor,
    ) -> None:
        """accumulate one env step. `step_term_rewards` is reward_manager._step_reward,
        shape (num_envs, num_terms); `done` is a float mask, shape (num_envs,)."""
        self.current_rewards += reward
        self.current_terms += step_term_rewards

        # fold finished episodes into the per-rollout aggregates (masked, so envs
        # that didn't finish contribute nothing).
        done_bool = done.bool()
        self.ep_sum += (self.current_rewards * done).sum()
        self.ep_sumsq += (self.current_rewards.square() * done).sum()
        self.ep_count += done.sum()
        self.term_sums += (self.current_terms * done.unsqueeze(-1)).sum(dim=0)
        self.ep_max = torch.maximum(
            self.ep_max,
            self.current_rewards.masked_fill(~done_bool, float("-inf")).max(),
        )
        self.ep_min = torch.minimum(
            self.ep_min,
            self.current_rewards.masked_fill(~done_bool, float("inf")).min(),
        )

        # zero only the envs that finished, so each starts its next episode at 0.
        not_done = 1.0 - done
        self.current_rewards *= not_done
        self.current_terms *= not_done.unsqueeze(-1)

    def summarize(self, entropies: torch.Tensor) -> RolloutStats:
        """reduce the episodes that finished this rollout into stats, then reset the
        accumulators"""
        n = int(self.ep_count.item())

        if n == 0:
            # forward-fill reward stats, but report zero completed episodes.
            self.last.num_episodes = 0
            self.reset_accumulators()
            return self.last

        mean = self.ep_sum / self.ep_count
        if n > 1:
            # sample variance
            var = (self.ep_sumsq - self.ep_count * mean.square()) / (self.ep_count - 1)
            std = var.clamp_min(0.0).sqrt().item()
        else:
            std = 0.0

        self.last.avg_reward = mean.item()
        self.last.min_reward = self.ep_min.item()
        self.last.max_reward = self.ep_max.item()
        self.last.std_reward = std
        self.last.avg_entropy = entropies.mean().item()
        self.last.num_episodes = n
        for i, name in enumerate(self.term_names):
            self.last.term_rewards[name] = (self.term_sums[i] / self.ep_count).item()

        self.reset_accumulators()
        return self.last
