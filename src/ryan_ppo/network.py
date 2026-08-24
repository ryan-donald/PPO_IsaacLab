from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ryan_ppo.normalization import ObsNormalization


class MLP(nn.Module):
    """
    class that defines the underlying network structure for both actor and critic,
    preventing needless repetition.
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        normalization_object: ObsNormalization | None,
        output_gain: float,
    ) -> None:
        super().__init__()

        self.obs_normalizer = normalization_object

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.output_layer.weight, gain=output_gain)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer:
            x = self.obs_normalizer(x)

        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        return self.output_layer(x)

    def update_normalization(self, obs: torch.Tensor) -> None:
        # update observation normalization statistics
        if self.obs_normalizer:
            self.obs_normalizer.update(obs)


class Actor(MLP):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        normalization_object: ObsNormalization | None = None,
        std: float = 0.5,
    ) -> None:
        super().__init__(
            state_dim, action_dim, hidden_dims, normalization_object, output_gain=0.01
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(std))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = super().forward(x)
        return mu, torch.exp(self.log_std), self.log_std


class Critic(MLP):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        normalization_object: ObsNormalization | None = None,
    ) -> None:
        super().__init__(
            state_dim, 1, hidden_dims, normalization_object, output_gain=1.0
        )
