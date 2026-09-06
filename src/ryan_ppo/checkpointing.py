from __future__ import annotations

from pathlib import Path

import torch


class CheckpointSaver:
    """writes weights and resumable checkpoint bundles for one training run.

    owns the run's log directory and the best-reward watermark, so train() only
    has to say which iteration just finished and how it scored.
    """

    SNAPSHOT_EVERY = 100

    def __init__(self, log_path: str, checkpoint_iters: str) -> None:
        self.log_path = Path(log_path)
        # iterations at which to save full resumable checkpoint bundles
        self.checkpoint_iters = {
            int(it) for it in checkpoint_iters.split(",") if it.strip()
        }
        self.best_reward = -float("inf")
        self.log_path.mkdir(parents=True, exist_ok=True)

    def _save_weights(self, agent, suffix: str) -> None:
        torch.save(agent.actor.state_dict(), self.log_path / f"actor_{suffix}.pth")
        torch.save(agent.critic.state_dict(), self.log_path / f"critic_{suffix}.pth")

    def save_iteration(
        self, agent, iteration: int, avg_reward: float, *, num_episodes: int
    ) -> None:
        # saves checkpoints if this iteration requires it. i.e. best reward, periodic.
        if num_episodes > 0 and avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self._save_weights(agent, "best")

        # periodic weight snapshots and a rolling resumable checkpoint
        if iteration % self.SNAPSHOT_EVERY == 0:
            self._save_weights(agent, f"iter_{iteration}")
            agent.save_checkpoint(self.log_path / "checkpoint_latest.pth", iteration)

        # full checkpoint bundles at requested iterations, for resuming or
        # fine-tuning later
        if iteration in self.checkpoint_iters:
            agent.save_checkpoint(
                self.log_path / f"checkpoint_{iteration}.pth", iteration
            )

    def save_final(self, agent, iteration: int) -> None:
        self._save_weights(agent, "final")
        agent.save_checkpoint(self.log_path / "checkpoint_final.pth", iteration)
