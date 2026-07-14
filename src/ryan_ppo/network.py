from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ryan_ppo.normalization import ObsNormalization

# bounds for the policy log-std. the mu for each joint is tanh activated, bounding
# them to [-1, 1]. this clamping prevents the log-std from becoming too large.
LOG_STD_MIN = np.log(0.005)
LOG_STD_MAX = np.log(1.0)

# currently training with a binary gripper action, and I am using this to prevent
# std collapsing very far from the mean for the gripper.
# I don't think this is necessary, but I am using it for testing currently.
GRIPPER_LOG_STD_MIN = np.log(0.3)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        normalization_object: ObsNormalization | None = None,
        std: float = 0.5,
    ) -> None:
        super().__init__()

        self.obs_normalizer = normalization_object

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # normalize observations, then forward pass
        if self.obs_normalizer:
            x = self.obs_normalizer(x)

        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        # pre-tanh values are returned, used to calculate the saturation loss in
        # update.
        pre_tanh = self.output_layer(x)
        mu = torch.tanh(pre_tanh)
        std = torch.exp(self.log_std)
        return mu, std, pre_tanh

    def update_normalization(self, obs: torch.Tensor) -> None:
        # update observation normalization statistics
        if self.obs_normalizer:
            self.obs_normalizer.update(obs)


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        normalization_object: ObsNormalization | None = None,
    ) -> None:
        super().__init__()

        self.obs_normalizer = normalization_object

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer:
            x = self.obs_normalizer(x)

        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        x = self.output_layer(x)
        return x

    def update_normalization(self, obs: torch.Tensor) -> None:
        # update observation normalization statistics
        if self.obs_normalizer:
            self.obs_normalizer.update(obs)
