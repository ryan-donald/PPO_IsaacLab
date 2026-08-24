from __future__ import annotations

import os

import torch


class CheckpointSaver:
    """writes weights and resumable checkpoint bundles for one training run.

    owns the run's log directory and the best-reward watermark, so train() only
    has to say which iteration just finished and how it scored.
    """

    SNAPSHOT_EVERY = 100

    def __init__(self, log_path: str, checkpoint_iters: str) -> None:
        self.log_path = log_path
        # iterations at which to save full resumable checkpoint bundles
        self.checkpoint_iters = {
            int(it) for it in checkpoint_iters.split(",") if it.strip()
        }
        self.best_reward = -float("inf")
        os.makedirs(log_path, exist_ok=True)

    def _save_weights(self, agent, suffix: str) -> None:
        torch.save(agent.actor.state_dict(), f"{self.log_path}actor_{suffix}.pth")
        torch.save(agent.critic.state_dict(), f"{self.log_path}critic_{suffix}.pth")

    def save_iteration(self, agent, iteration: int, avg_reward: float) -> None:
        # saves checkpoints if this iteration requires it. i.e. best reward, periodic.
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self._save_weights(agent, "best")

        # periodic weight snapshots and a rolling resumable checkpoint
        if iteration % self.SNAPSHOT_EVERY == 0:
            self._save_weights(agent, f"iter_{iteration}")
            agent.save_checkpoint(f"{self.log_path}checkpoint_latest.pth", iteration)

        # full checkpoint bundles at requested iterations, for resuming or
        # fine-tuning later
        if iteration in self.checkpoint_iters:
            agent.save_checkpoint(
                f"{self.log_path}checkpoint_{iteration}.pth", iteration
            )

    def save_final(self, agent, iteration: int) -> None:
        self._save_weights(agent, "final")
        agent.save_checkpoint(f"{self.log_path}checkpoint_final.pth", iteration)
