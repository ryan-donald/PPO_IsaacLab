import torch

from ryan_ppo.ppo import PPOAgent


def test_agent_init():
    # tests that the agent creates the two networks and they have the output shape.
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    batch_size = 8

    agent = PPOAgent(state_dim, action_dim, hidden_dims=hidden_dims)

    random_input = torch.randn(batch_size, state_dim)

    critic_output = agent.critic.forward(random_input)
    mu, std = agent.actor.forward(random_input)

    assert critic_output.shape == (batch_size, 1), (
        "Critic output should be (batch_size, 1)"
    )

    assert critic_output.requires_grad, "Critic output should require grad"

    assert mu.shape == (batch_size, action_dim), (
        "Actor output mu should be (batch_size, action_dim)"
    )
    assert std.shape == (action_dim,), "Actor output std should be (action_dim)"

    assert mu.requires_grad, "Actor output should require grad"


def test_select_action():
    # tests the method for selecting an action,
    # checks that output shape matches what is expected
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    batch_size = 8

    agent = PPOAgent(state_dim, action_dim, hidden_dims=hidden_dims)

    random_input = torch.randn(batch_size, state_dim)

    action, log_prob, entropy = agent.select_action(random_input)

    assert action.shape == (batch_size, action_dim), (
        "Action should be in shape (batch_size, action_dim)"
    )
    assert log_prob.shape == (batch_size,), "Log_prob should be in shape (batch_size,)"
    assert entropy.shape == (batch_size,), "Entropy should be in shape (batch_size,)"


def test_compute_gae():
    # tests the compute_gae method, ensure correct shape and data with basic input
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    num_steps = 4
    num_envs = 2

    agent = PPOAgent(state_dim, action_dim, hidden_dims=hidden_dims)

    random_rewards = torch.tensor(
        [[1.1000, 0.7000], [0.7000, 0.1000], [0.0000, 0.0000], [0.2000, -0.9000]]
    )
    random_values = torch.tensor(
        [[1.2000, -0.1000], [-0.0000, 0.1000], [-0.5000, 0.5000], [1.3000, -0.7000]]
    )
    random_dones = torch.tensor(
        [[-0.3000, 0.3000], [-0.5000, -0.0000], [-0.5000, -1.3000], [0.8000, 1.9000]]
    )
    random_next_value = torch.tensor([-0.4000, 1.4000])

    advantages, returns = agent.compute_gae(
        random_rewards, random_values, random_dones, random_next_value
    )

    assert advantages.shape == (num_steps, num_envs), (
        "advantages should be (num_steps, num_envs)"
    )
    assert returns.shape == (num_steps, num_envs), (
        "returns should be (num_steps, num_envs)"
    )

    assert torch.allclose(
        advantages,
        torch.tensor(
            [
                [1.1709, -2.0399],
                [1.0395, -4.4190],
                [0.7669, -5.2248],
                [-1.1792, -1.4474],
            ]
        ),
        rtol=1e-4,
        atol=1e-4,
    ), "advantages are wrong"

    assert torch.allclose(
        returns,
        torch.tensor(
            [[2.3709, -2.1399], [1.0395, -4.3190], [0.2669, -4.7248], [0.1208, -2.1474]]
        ),
        rtol=1e-4,
        atol=1e-4,
    ), "returns are wrong"


def test_update():
    # tests that the update function runs and changes the weights.
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    batch_size = 8

    agent = PPOAgent(state_dim, action_dim, hidden_dims=hidden_dims)

    random_states = torch.randn(batch_size, state_dim)
    random_actions = torch.randn(batch_size, action_dim)
    # log_probs, returns, advantages, values_old should be 1D depending on your batching
    random_log_probs_old = torch.randn(batch_size)
    random_returns = torch.randn(batch_size)
    random_advantages = torch.randn(batch_size)
    random_values_old = torch.randn(batch_size)
    random_mus_old = torch.randn(batch_size, action_dim)
    random_stds_old = torch.randn(batch_size, action_dim)
    epochs = 4

    actor_old_params = [p.clone() for p in agent.actor.parameters()]
    critic_old_params = [p.clone() for p in agent.critic.parameters()]

    kl = agent.update(
        random_states,
        random_actions,
        random_log_probs_old,
        random_returns,
        random_advantages,
        random_values_old,
        random_mus_old,
        random_stds_old,
        epochs,
        batch_size,
    )

    assert type(kl) is float

    actor_new_params = [p.clone() for p in agent.actor.parameters()]
    critic_new_params = [p.clone() for p in agent.critic.parameters()]

    assert any(
        not torch.allclose(old, new)
        for old, new in zip(actor_old_params, actor_new_params)
    ), "actor weights did not update"
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(critic_old_params, critic_new_params)
    ), "critic weights did not update"
