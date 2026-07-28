import torch
import torch.nn as nn

from ryan_ppo.network import Actor, Critic


def test_actor_init():
    # testing that the network is created and has parameters
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    actor = Actor(state_dim, action_dim, hidden_dims)

    total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    assert total_params > 0


def test_actor_forward():
    # testing that the network is able to take in data in the correct shape
    # and output in the correct shape
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    batch_size = 8

    actor = Actor(state_dim, action_dim, hidden_dims)

    random_input = torch.randn(batch_size, state_dim)

    mu, std = actor(random_input)

    assert mu.shape == (batch_size, action_dim)
    assert std.shape == (action_dim,)

    assert mu.requires_grad


def test_actor_weights_init():
    # checks that the weights and biases are initialized correctly.
    state_dim = 4
    action_dim = 4
    hidden_dims = [2, 2]
    actor = Actor(state_dim, action_dim, hidden_dims)

    for name, module in actor.named_modules():
        if name != "output_layer" and isinstance(module, nn.Linear):
            assert torch.allclose(
                torch.linalg.norm(module.weight, ord=2),
                torch.sqrt(torch.tensor(2)),
                rtol=1e-4,
                atol=1e-4,
            )
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

        elif name == "output_layer" and isinstance(module, nn.Linear):
            assert torch.allclose(
                torch.linalg.norm(module.weight, ord=2),
                torch.tensor(0.01),
                rtol=1e-4,
                atol=1e-4,
            )
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))


def test_critic_init():
    # testing that the network is created and has parameters
    state_dim = 4
    hidden_dims = [2, 2]
    critic = Critic(state_dim, hidden_dims)

    total_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    assert total_params > 0


def test_critic_forward():
    # testing that the network is able to take in data in the correct shape
    # and output in the correct shape
    state_dim = 4
    hidden_dims = [2, 2]
    batch_size = 8

    critic = Critic(state_dim, hidden_dims)

    random_input = torch.randn(batch_size, state_dim)

    output = critic(random_input)

    assert output.shape == (batch_size, 1)

    assert output.requires_grad


def test_critic_weights_init():
    # checks that the weights are initialized to non-zero values,
    # and that the bias' are initialized to zero.
    state_dim = 4
    hidden_dims = [2, 2]
    critic = Critic(state_dim, hidden_dims)

    for name, module in critic.named_modules():
        if name != "output_layer" and isinstance(module, nn.Linear):
            assert torch.allclose(
                torch.linalg.norm(module.weight, ord=2),
                torch.sqrt(torch.tensor(2)),
                rtol=1e-4,
                atol=1e-4,
            )
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

        elif name == "output_layer" and isinstance(module, nn.Linear):
            assert torch.allclose(
                torch.linalg.norm(module.weight, ord=2),
                torch.tensor(1.0),
                rtol=1e-4,
                atol=1e-4,
            )
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
