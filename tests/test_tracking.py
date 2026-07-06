import pytest
import torch

from ryan_ppo.tracking import EpisodeTracker


def make_tracker(num_envs=2, terms=("a", "b")):
    return EpisodeTracker(num_envs, terms, torch.device("cpu"))


def test_completed_episode_stats():
    tracker = make_tracker()
    # per-step term rewards: env 0 gets (0.5, 0.5), env 1 gets (1.0, 1.0)
    term_r = torch.tensor([[0.5, 0.5], [1.0, 1.0]])

    # step 1: no episode ends; step 2: env 0 finishes with total reward 2.0
    tracker.record_step(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), term_r)
    tracker.record_step(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]), term_r)

    stats = tracker.summarize(entropies=torch.zeros(2, 2))
    assert stats.num_episodes == 1
    assert stats.avg_reward == pytest.approx(2.0)
    assert stats.min_reward == pytest.approx(2.0)
    assert stats.max_reward == pytest.approx(2.0)
    assert stats.term_rewards["a"] == pytest.approx(1.0)
    assert stats.term_rewards["b"] == pytest.approx(1.0)


def test_episode_spanning_rollouts():
    tracker = make_tracker(num_envs=1, terms=("a",))
    term_r = torch.tensor([[1.0]])

    # rollout 1: the episode does not finish, so its reward keeps accumulating
    tracker.record_step(torch.tensor([1.5]), torch.tensor([0.0]), term_r)
    stats = tracker.summarize(entropies=torch.zeros(1, 1))
    assert stats.num_episodes == 0

    # rollout 2: the episode finishes; its return spans both rollouts
    tracker.record_step(torch.tensor([2.5]), torch.tensor([1.0]), term_r)
    stats = tracker.summarize(entropies=torch.zeros(1, 1))
    assert stats.num_episodes == 1
    assert stats.avg_reward == pytest.approx(4.0)
    assert stats.term_rewards["a"] == pytest.approx(2.0)


def test_empty_rollout_forward_fills():
    tracker = make_tracker(num_envs=1, terms=("a",))
    term_r = torch.tensor([[1.0]])

    tracker.record_step(torch.tensor([3.0]), torch.tensor([1.0]), term_r)
    stats = tracker.summarize(entropies=torch.zeros(1, 1))
    assert stats.num_episodes == 1
    assert stats.avg_reward == pytest.approx(3.0)

    # nothing finishes this rollout: reward stats forward-fill from the previous
    # rollout, but the episode count honestly reports zero.
    stats = tracker.summarize(entropies=torch.zeros(1, 1))
    assert stats.num_episodes == 0
    assert stats.avg_reward == pytest.approx(3.0)
