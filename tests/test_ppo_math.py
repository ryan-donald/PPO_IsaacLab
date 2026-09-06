"""Pins the hand-rolled gaussian math against torch.distributions.

ppo.py computes log-probs, entropy and KL by hand because
torch.distributions.Normal is far slower in the rollout hot path. These tests
keep the fast versions honest: a sign or factor error here does not crash, it
just quietly trains a worse policy.
"""

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from ryan_ppo.config import TrainConfig
from ryan_ppo.ppo import PPOAgent, gaussian_entropy, gaussian_log_prob

BATCH, ACTION_DIM = 64, 6


def make_agent(state_dim=8, action_dim=ACTION_DIM, hidden_dims=(16, 16)):
    cfg = TrainConfig(
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=1.0,
        clip_epsilon=0.2,
        max_grad_norm=1.0,
        desired_kl=0.01,
        entropy_coef=0.01,
        schedule_type="adaptive",
        num_learning_epochs=1,
        num_steps_per_env=8,
        num_mini_batches=1,
        max_iterations=1,
        use_normalization=True,
        hidden_dims=list(hidden_dims),
    )
    return PPOAgent(state_dim, action_dim, cfg)


def test_gaussian_log_prob_matches_torch():
    torch.manual_seed(0)
    mu = torch.randn(BATCH, ACTION_DIM)
    log_std = torch.randn(ACTION_DIM) * 0.5
    std = log_std.exp()
    x = torch.randn(BATCH, ACTION_DIM)

    expected = Normal(mu, std).log_prob(x).sum(dim=-1)
    assert torch.allclose(
        gaussian_log_prob(x, mu, std, log_std), expected, rtol=1e-5, atol=1e-6
    )


def test_gaussian_entropy_matches_torch():
    torch.manual_seed(1)
    log_std = torch.randn(ACTION_DIM) * 0.5
    expected = Normal(torch.zeros(ACTION_DIM), log_std.exp()).entropy().sum()
    assert torch.allclose(gaussian_entropy(log_std), expected, rtol=1e-5, atol=1e-6)


def test_agent_entropy_matches_torch():
    agent = make_agent()
    log_std = agent.actor.log_std.detach()
    expected = Normal(torch.zeros_like(log_std), log_std.exp()).entropy().sum()
    assert abs(agent.entropy() - expected.item()) < 1e-5


def test_act_log_prob_matches_its_own_distribution():
    # the sampled action and the log-prob returned alongside it must agree,
    # through the compiled path that training actually uses.
    torch.manual_seed(2)
    agent = make_agent()
    states = torch.randn(BATCH, 8)

    action, log_prob, mu, std = agent.select_action(states)

    expected = Normal(mu, std).log_prob(action).sum(dim=-1)
    assert torch.allclose(log_prob, expected, rtol=1e-5, atol=1e-5)


def test_minibatch_kl_matches_torch():
    # minibatch_loss returns the batch-mean KL(old || new). the closed form in
    # ppo.py splits the std terms out of the batch mean, which is only valid
    # because std is state-independent -- this checks the two agree.
    torch.manual_seed(3)
    agent = make_agent()
    states = torch.randn(BATCH, 8)

    with torch.no_grad():
        mu_old, std_old, _ = agent.actor(states)
    mu_old = mu_old + 0.1 * torch.randn_like(mu_old)  # make old != new
    std_old = std_old * 1.3
    actions = torch.randn(BATCH, ACTION_DIM)

    _, kl = agent.minibatch_loss(
        states,
        actions,
        torch.randn(BATCH),
        torch.randn(BATCH),
        torch.randn(BATCH),
        torch.randn(BATCH),
        mu_old,
        std_old,
        torch.arange(BATCH),
    )

    with torch.no_grad():
        mu_new, std_new, _ = agent.actor(states)
    expected = (
        kl_divergence(Normal(mu_old, std_old), Normal(mu_new, std_new))
        .sum(dim=-1)
        .mean()
    )
    assert torch.allclose(kl.detach(), expected, rtol=1e-4, atol=1e-6), (
        kl.item(),
        expected.item(),
    )


def test_minibatch_kl_is_zero_for_identical_policies():
    torch.manual_seed(4)
    agent = make_agent()
    states = torch.randn(BATCH, 8)

    with torch.no_grad():
        mu, std, _ = agent.actor(states)

    _, kl = agent.minibatch_loss(
        states,
        torch.randn(BATCH, ACTION_DIM),
        torch.randn(BATCH),
        torch.randn(BATCH),
        torch.randn(BATCH),
        torch.randn(BATCH),
        mu,
        std,
        torch.arange(BATCH),
    )
    assert abs(kl.item()) < 1e-6, kl.item()


@pytest.mark.parametrize("sigma", [1.0, 0.005, 1e-4, 1e-6])
@pytest.mark.parametrize("identical", [False, True])
def test_kl_at_small_standard_deviations(sigma, identical):
    agent = make_agent()
    states = torch.zeros(BATCH, 8)
    with torch.no_grad():
        agent.actor.log_std.fill_(torch.tensor(sigma).log())
        mu, std, _ = agent.actor(states)
    old_mu = mu if identical else mu + sigma * 0.25
    old_std = std if identical else std * 1.3
    _, actual = agent.minibatch_loss(
        states,
        mu,
        torch.zeros(BATCH),
        torch.zeros(BATCH),
        torch.ones(BATCH),
        torch.zeros(BATCH),
        old_mu,
        old_std,
        torch.arange(BATCH),
    )
    expected = (
        kl_divergence(
            Normal(old_mu.double(), old_std.double()), Normal(mu.double(), std.double())
        )
        .sum(-1)
        .mean()
    )
    assert actual.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)
