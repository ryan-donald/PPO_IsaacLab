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

    stats = tracker.summarize(avg_entropy=0.0)
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
    stats = tracker.summarize(avg_entropy=0.0)
    assert stats.num_episodes == 0

    # rollout 2: the episode finishes; its return spans both rollouts
    tracker.record_step(torch.tensor([2.5]), torch.tensor([1.0]), term_r)
    stats = tracker.summarize(avg_entropy=0.0)
    assert stats.num_episodes == 1
    assert stats.avg_reward == pytest.approx(4.0)
    assert stats.term_rewards["a"] == pytest.approx(2.0)


def test_all_completions_counted():
    # every episode finishing in the rollout contributes, not a fixed-size sample of
    # them. env i finishes with return i, so the mean pins the whole population.
    num_envs = 500
    tracker = make_tracker(num_envs=num_envs, terms=("a",))
    reward = torch.arange(num_envs, dtype=torch.float)

    tracker.record_step(reward, torch.ones(num_envs), reward.unsqueeze(-1))
    stats = tracker.summarize(avg_entropy=0.0)

    assert stats.num_episodes == num_envs
    assert stats.avg_reward == pytest.approx((num_envs - 1) / 2)
    assert stats.min_reward == pytest.approx(0.0)
    assert stats.max_reward == pytest.approx(num_envs - 1)
    assert stats.term_rewards["a"] == pytest.approx(stats.avg_reward)


def test_empty_rollout_forward_fills():
    tracker = make_tracker(num_envs=1, terms=("a",))
    term_r = torch.tensor([[1.0]])

    tracker.record_step(torch.tensor([3.0]), torch.tensor([1.0]), term_r)
    stats = tracker.summarize(avg_entropy=0.0)
    assert stats.num_episodes == 1
    assert stats.avg_reward == pytest.approx(3.0)

    # nothing finishes this rollout: reward stats forward-fill from the previous
    # rollout, but the episode count honestly reports zero.
    stats = tracker.summarize(avg_entropy=0.0)
    assert stats.num_episodes == 0
    assert stats.avg_reward == pytest.approx(3.0)
