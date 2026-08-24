import numpy as np
import torch
import torch.nn as nn

from ryan_ppo.network import Actor, Critic


def assert_orthogonal_init(net, output_gain):
    # hidden layers use gain sqrt(2), the output layer the gain given; all biases
    # are zeroed.
    for layer in net.hidden_layers:
        assert isinstance(layer, nn.Linear)
        assert torch.allclose(
            torch.linalg.norm(layer.weight, ord=2),
            torch.sqrt(torch.tensor(2.0)),
            rtol=1e-4,
            atol=1e-4,
        )
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    assert torch.allclose(
        torch.linalg.norm(net.output_layer.weight, ord=2),
        torch.tensor(output_gain),
        rtol=1e-4,
        atol=1e-4,
    )
    assert torch.allclose(
        net.output_layer.bias, torch.zeros_like(net.output_layer.bias)
    )


def test_actor_init():
    # testing that the network is created and has parameters
    actor = Actor(4, 4, [2, 2])

    total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    assert total_params > 0


def test_actor_forward():
    # testing that the network is able to take in data in the correct shape
    # and output in the correct shape
    state_dim = 4
    action_dim = 4
    batch_size = 8

    actor = Actor(state_dim, action_dim, [2, 2])

    mu, std, log_std = actor(torch.randn(batch_size, state_dim))

    assert mu.shape == (batch_size, action_dim)
    assert std.shape == (action_dim,)
    assert log_std.shape == (action_dim,)

    # std and log_std describe the same distribution.
    assert torch.allclose(std, torch.exp(log_std))

    assert mu.requires_grad


def test_actor_default_std():
    # log_std starts at log(std) for every action dim.
    actor = Actor(4, 3, [2, 2], std=0.5)
    assert torch.allclose(
        actor.log_std, torch.full((3,), float(np.log(0.5))), rtol=1e-6, atol=1e-6
    )


def test_actor_weights_init():
    # checks that the weights and biases are initialized correctly.
    assert_orthogonal_init(Actor(4, 4, [2, 2]), output_gain=0.01)


def test_critic_init():
    # testing that the network is created and has parameters
    critic = Critic(4, [2, 2])

    total_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    assert total_params > 0


def test_critic_forward():
    # testing that the network is able to take in data in the correct shape
    # and output in the correct shape
    state_dim = 4
    batch_size = 8

    critic = Critic(state_dim, [2, 2])

    output = critic(torch.randn(batch_size, state_dim))

    assert output.shape == (batch_size, 1)

    assert output.requires_grad


def test_critic_weights_init():
    # checks that the weights are initialized to non-zero values,
    # and that the bias' are initialized to zero.
    assert_orthogonal_init(Critic(4, [2, 2]), output_gain=1.0)
