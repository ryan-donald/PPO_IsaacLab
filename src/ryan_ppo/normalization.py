import torch
from torch import nn


class ObsNormalization(nn.Module):
    """
    Welford's Algorithm for Online calculation of
    Mean and Variance for a distribution.
    """

    def __init__(self, state_dim: int, epsilon: float = 1e-2):
        super().__init__()

        self.epsilon = epsilon

        self.register_buffer("mean", torch.zeros(state_dim))
        self.register_buffer("var", torch.ones(state_dim))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32))

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta_mean = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta_mean * batch_count / total_count

        self.var.mul_(self.count).add_(batch_var * batch_count).add_(
            torch.square(delta_mean) * (self.count * batch_count / total_count)
        ).div_(total_count)
        self.count.copy_(total_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var) + self.epsilon
        return (x - self.mean) / std
