from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import wandb
from rich.live import Live

from ryan_ppo.utils import generate_table


@dataclass
class RolloutStats:
    """
    reduced episode statistics for one rollout, consumed by the logger.
    """

    avg_reward: float
    min_reward: float
    max_reward: float
    std_reward: float
    avg_entropy: float
    num_episodes: int
    term_rewards: dict[str, float]


class EpisodeTracker:
    """
    accumulates per-env episode returns and per-term rewards, then reduces over
    every episode that finished during the rollout.
    """

    def __init__(
        self,
        num_envs: int,
        term_names,
        device: torch.device,
    ) -> None:
        self.term_names = list(term_names)
        self.num_terms = len(self.term_names)

        # per-env running returns, kept across rollouts so episodes longer than one
        # rollout still accumulate correctly.
        self.current_rewards = torch.zeros(num_envs, device=device)
        self.current_terms = torch.zeros(num_envs, self.num_terms, device=device)

        # totals since the last summarize(). float64 so the sum-of-squares variance
        # stays accurate; they are scalars, so the precision is free.
        self.device = device
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.ret_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.ret_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.term_sum = torch.zeros(self.num_terms, dtype=torch.float64, device=device)
        self.ret_min = torch.full((), float("inf"), dtype=torch.float64, device=device)
        self.ret_max = torch.full((), -float("inf"), dtype=torch.float64, device=device)

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
        """
        accumulate one env step. `done` is a float mask over envs, and the
        per-term rewards are scaled so they sum to `reward`.
        """
        self.current_rewards += reward
        self.current_terms += step_term_rewards

        # fold in the envs that finished. the mask zeroes the rest, so they add
        # nothing to the totals.
        done_bool = done.bool()
        finished = self.current_rewards * done
        self.count += done.sum()
        self.ret_sum += finished.sum()
        self.ret_sq_sum += finished.square().sum()
        self.term_sum += (self.current_terms * done.unsqueeze(-1)).sum(0)

        inf = float("inf")
        self.ret_min.copy_(
            torch.minimum(
                self.ret_min, torch.where(done_bool, self.current_rewards, inf).min()
            )
        )
        self.ret_max.copy_(
            torch.maximum(
                self.ret_max, torch.where(done_bool, self.current_rewards, -inf).max()
            )
        )

        # zero only the finished envs, so each starts its next episode at 0.
        not_done = 1.0 - done
        self.current_rewards *= not_done
        self.current_terms *= not_done.unsqueeze(-1)

    @torch.compile
    def record_rollout(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        term_rewards: torch.Tensor,
    ) -> None:
        """Accumulate a fixed-length rollout in one compiled graph."""
        for step in range(rewards.shape[0]):
            self.record_step(rewards[step], dones[step], term_rewards[step])

    def summarize(self, avg_entropy: float) -> RolloutStats:
        """
        reduce the episodes finished since the last call into stats.
        """
        self.last.avg_entropy = avg_entropy

        # one host transfer for every accumulator rather than one per field.
        totals = torch.cat(
            [
                self.count.view(1),
                self.ret_sum.view(1),
                self.ret_sq_sum.view(1),
                self.ret_min.view(1),
                self.ret_max.view(1),
                self.term_sum,
            ]
        ).tolist()

        self.count.zero_()
        self.ret_sum.zero_()
        self.ret_sq_sum.zero_()
        self.term_sum.zero_()
        self.ret_min.fill_(float("inf"))
        self.ret_max.fill_(-float("inf"))

        n = int(totals[0])
        self.last.num_episodes = n
        if n == 0:
            # nothing finished this rollout; forward-fill the previous stats.
            return self.last

        ret_sum, ret_sq_sum, ret_min, ret_max = totals[1:5]
        mean = ret_sum / n
        if n > 1:
            # sample variance
            var = (ret_sq_sum - n * mean * mean) / (n - 1)
            std = max(var, 0.0) ** 0.5
        else:
            std = 0.0

        self.last.avg_reward = mean
        self.last.min_reward = ret_min
        self.last.max_reward = ret_max
        self.last.std_reward = std
        for i, name in enumerate(self.term_names):
            self.last.term_rewards[name] = totals[5 + i] / n

        return self.last


class TrainingLogger:
    """
    owns the wandb stream and the live terminal table for one training run,
    keeping the display out of train.py.
    """

    def __init__(
        self,
        task: str,
        run_url: str | None,
        term_names,
        max_iterations: int,
        steps_per_rollout: int,
    ) -> None:
        self.task = task
        self.run_url = run_url
        self.term_names = list(term_names)
        self.max_iterations = max_iterations
        self.steps_per_rollout = steps_per_rollout

        self.perf_stats = {
            "steps": 0,
            "steps/s": 0.0,
            "Rollout Time": 0.0,
            "Preparation Time": 0.0,
            "Update Time": 0.0,
            "episodes": 0.0,
            "Runtime": 0.0,
            "Remaining Time": 0.0,
        }
        self.train_stats = {
            "lr": 0.0,
            "kl": 0.0,
            "entropy": 0.0,
            "Iteration": 0,
        }

        self.start_time = time.perf_counter()
        self.live = Live(
            self._table({name: 0.0 for name in self.term_names}),
            refresh_per_second=4,
        )

    def _table(self, reward_rows: dict[str, float]):
        return generate_table(
            self.perf_stats, self.train_stats, reward_rows, self.task, self.run_url
        )

    def start(self) -> None:
        self.live.start()

    def stop(self) -> None:
        self.live.stop()

    def log_iteration(
        self,
        iteration: int,
        stats: RolloutStats,
        mean_kl: float,
        lr: float,
        rollout_time: float,
        update_time: float,
        *,
        preparation_time: float = 0.0,
        warmup: bool = False,
    ) -> None:
        """
        log one iteration to wandb and refresh the terminal table. `iteration`
        is zero-based for wandb; the table shows it one-based.
        """
        logging_dict = {
            "train/avg_reward": stats.avg_reward,
            "train/min_reward": stats.min_reward,
            "train/max_reward": stats.max_reward,
            "train/std_reward": stats.std_reward,
            "train/kl": mean_kl,
            "train/lr": lr,
            "train/episodes": stats.num_episodes,
            "train/avg_entropy": stats.avg_entropy,
            "perf/rollout_time": rollout_time,
            "perf/update_time": update_time,
            "perf/preparation_time": preparation_time,
            "perf/warmup": warmup,
        }
        for name in self.term_names:
            logging_dict[f"rewards/{name}"] = stats.term_rewards[name]

        wandb.log(logging_dict, step=iteration)

        # aggregate rows for the TUI, added after wandb.log so they aren't logged.
        reward_rows = dict(stats.term_rewards)
        reward_rows["Mean Reward"] = stats.avg_reward
        reward_rows["Max Reward"] = stats.max_reward

        perf = self.perf_stats
        perf["steps"] += self.steps_per_rollout
        perf["Runtime"] = time.perf_counter() - self.start_time
        perf["steps/s"] = perf["steps"] / perf["Runtime"]
        perf["Rollout Time"] = rollout_time
        perf["Preparation Time"] = preparation_time
        perf["Update Time"] = update_time
        perf["episodes"] += stats.num_episodes
        perf["Remaining Time"] = (
            (self.max_iterations - (iteration + 1))
            * self.steps_per_rollout
            / perf["steps/s"]
        )

        self.train_stats["lr"] = lr
        self.train_stats["kl"] = mean_kl
        self.train_stats["entropy"] = stats.avg_entropy
        self.train_stats["Iteration"] = iteration + 1

        self.live.update(self._table(reward_rows))
