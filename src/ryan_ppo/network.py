from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from rsl_rl.modules import EmpiricalNormalization


class Actor(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list[int] = [64, 64],
                 normalization_object: EmpiricalNormalization = None,
                 std: float = 1.0) -> None:
        super(Actor, self).__init__()

        if normalization_object:
            self.obs_normalizer = normalization_object
        else:
            self.obs_normalizer = None

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(std))

        self._init_weights()

    def _init_weights(self) -> None:
        # orthogonal initialization for all layers. output layer has small gain 0.01
        for name, module in self.named_modules():
            if name != "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif name == "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        # normalize observations, then forward pass
        if self.obs_normalizer:
            x = self.obs_normalizer(x)

        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        mu = self.output_layer(x)
        std = torch.exp(self.log_std)
        return mu, std

    def update_normalization(self, obs: torch.tensor) -> None:
        # update observation normalization statistics
        if self.obs_normalizer:
            self.obs_normalizer.update(obs)


class Critic(nn.Module):

    def __init__(self,
                 state_dim: int,
                 hidden_dims: list[int] = [64, 64],
                 normalization_object: EmpiricalNormalization = None) -> None:
        super(Critic, self).__init__()

        if normalization_object:
            self.obs_normalizer = normalization_object
        else:
            self.obs_normalizer = None

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # orthogonal initialization for all layers. output layer has gain 1
        for name, module in self.named_modules():
            if name != "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif name == "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.obs_normalizer:
            x = self.obs_normalizer(x)

        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        x = self.output_layer(x)
        return x

    def update_normalization(self, obs: torch.tensor) -> None:
        # update observation normalization statistics
        if self.obs_normalizer:
            self.obs_normalizer.update(obs)
