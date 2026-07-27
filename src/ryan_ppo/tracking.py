from __future__ import annotations

from dataclasses import dataclass

import torch

# allows for more accurate logging, rollouts with very few finished episodes can't tank
# the logged rewards now.
REWARD_WINDOW = 100


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
    then reduces over a ring buffer of the most recent REWARD_WINDOW episodes.

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

        # ring of the last REWARD_WINDOW completed episodes.
        self.window = REWARD_WINDOW
        self.ret_buf = torch.zeros(self.window + 1, device=device)
        self.term_buf = torch.zeros(self.window + 1, self.num_terms, device=device)
        self.scratch = torch.tensor(self.window, device=device)
        self.slot_idx = torch.arange(self.window, device=device)

        self.ptr = torch.zeros((), dtype=torch.long, device=device)
        self.filled = torch.zeros((), dtype=torch.long, device=device)
        self.rollout_count = torch.zeros((), dtype=torch.long, device=device)

        # mutated in place each summarize().
        self.last = RolloutStats(
            avg_reward=0.0,
            min_reward=0.0,
            max_reward=0.0,
            std_reward=0.0,
            avg_entropy=0.0,
            num_episodes=0,
            term_rewards={name: 0.0 for name in self.term_names},
        )

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

        # write finished episodes into the ring. cumsum gives each finished env a
        # distinct offset from the write pointer; unfinished envs go to the scratch
        # slot. if more than `window` finish on one step the modulo makes them
        # collide. This will store the 100 most recently finished envs, counting
        # backwards from the end of the list. i.e., the env at (num_envs-1) idx,
        # down to 0.
        done_bool = done.bool()
        done_long = done_bool.long()
        offsets = done_long.cumsum(0) - 1
        slot = torch.where(done_bool, (self.ptr + offsets) % self.window, self.scratch)
        self.ret_buf.index_put_((slot,), self.current_rewards)
        self.term_buf.index_put_((slot,), self.current_terms)

        num_done = done_long.sum()
        self.ptr = (self.ptr + num_done) % self.window
        self.filled = torch.clamp(self.filled + num_done, max=self.window)
        self.rollout_count += num_done

        # zero only the envs that finished, so each starts its next episode at 0.
        not_done = 1.0 - done
        self.current_rewards *= not_done
        self.current_terms *= not_done.unsqueeze(-1)

    def summarize(self, avg_entropy: float) -> RolloutStats:
        """reduce the last `window` completed episodes into stats."""
        self.last.avg_entropy = avg_entropy
        self.last.num_episodes = int(self.rollout_count.item())
        self.rollout_count.zero_()

        filled = int(self.filled.item())
        if filled == 0:
            # nothing has finished yet anywhere in the run; leave stats at zero.
            return self.last

        valid = self.slot_idx < self.filled
        rets = self.ret_buf[: self.window]
        n = self.filled.float()

        mean = (rets * valid).sum() / n
        if filled > 1:
            # sample variance
            sumsq = (rets.square() * valid).sum()
            var = (sumsq - n * mean.square()) / (n - 1)
            std = var.clamp_min(0.0).sqrt().item()
        else:
            std = 0.0

        term_means = (self.term_buf[: self.window] * valid.unsqueeze(-1)).sum(0) / n

        self.last.avg_reward = mean.item()
        self.last.min_reward = rets.masked_fill(~valid, float("inf")).min().item()
        self.last.max_reward = rets.masked_fill(~valid, float("-inf")).max().item()
        self.last.std_reward = std
        for i, name in enumerate(self.term_names):
            self.last.term_rewards[name] = term_means[i].item()

        return self.last
