from types import SimpleNamespace
from unittest.mock import Mock

import torch

from ryan_ppo.checkpointing import CheckpointSaver


def test_best_requires_completed_episodes_but_snapshots_do_not(tmp_path):
    agent = SimpleNamespace(
        actor=torch.nn.Linear(2, 1),
        critic=torch.nn.Linear(2, 1),
        save_checkpoint=Mock(),
    )
    saver = CheckpointSaver(str(tmp_path), "100")
    saver.save_iteration(agent, 1, 0.0, num_episodes=0)
    assert saver.best_reward == -float("inf")
    assert not (tmp_path / "actor_best.pth").exists()

    saver.save_iteration(agent, 2, -1.0, num_episodes=1)
    assert saver.best_reward == -1.0
    assert (tmp_path / "actor_best.pth").exists()

    saver.save_iteration(agent, 100, 0.0, num_episodes=0)
    assert saver.best_reward == -1.0
    assert (tmp_path / "actor_iter_100.pth").exists()
    agent.save_checkpoint.assert_any_call(tmp_path / "checkpoint_latest.pth", 100)
    agent.save_checkpoint.assert_any_call(tmp_path / "checkpoint_100.pth", 100)
