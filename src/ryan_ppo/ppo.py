from __future__ import annotations

import math
import os

import torch
import torch.optim as optim

from ryan_ppo.config import TrainConfig
from ryan_ppo.network import Actor, Critic
from ryan_ppo.normalization import ObsNormalization
from ryan_ppo.storage import RolloutBatch

KL_LR_DECREASE_FACTOR = 2.0
KL_LR_INCREASE_FACTOR = 2.0
LR_ADJUST_RATIO = 1.5

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)

# performance improvements by adjusting compile mode for minibatch loss.
COMPILE_MODE = os.environ.get("RYAN_PPO_COMPILE_MODE", "max-autotune-no-cudagraphs")
if COMPILE_MODE == "default":
    COMPILE_MODE = None


def strip_compile_prefix(state_dict: dict) -> dict:
    # allows loading of checkpoints saved from a compiled module.
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def gaussian_log_prob(
    x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    # diagonal gaussian log-density, summed over action dims.
    return (-0.5 * ((x - mu) / std).square() - log_std - LOG_SQRT_2PI).sum(dim=-1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    # closed-form diagonal gaussian entropy, summed over action dims.
    return (log_std + 0.5 + LOG_SQRT_2PI).sum()


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: TrainConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        if cfg.use_normalization:
            self.obs_normalizer = ObsNormalization(state_dim)
        else:
            self.obs_normalizer = None

        self.device = device

        self.actor = Actor(
            state_dim,
            action_dim,
            cfg.hidden_dims,
            self.obs_normalizer,
        ).to(device)
        self.critic = Critic(state_dim, cfg.hidden_dims, self.obs_normalizer).to(device)

        self.log_std_min = math.log(cfg.std_min)
        self.log_std_max = math.log(cfg.std_max)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        # learning rate stored as tensor for speed.
        self.lr_t = torch.tensor(float(cfg.learning_rate), device=device)

        self.optimizer = optim.Adam(
            self.actor_params + self.critic_params,
            lr=self.lr_t,
            fused=(device.type == "cuda"),
            foreach=False if device.type != "cuda" else None,
        )

        # hyperparameters
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_epsilon = cfg.clip_epsilon
        self.max_grad_norm = cfg.max_grad_norm
        self.entropy_coef = cfg.entropy_coef
        self.value_coef = cfg.value_coef
        self.desired_kl = cfg.desired_kl
        self.schedule_type = cfg.schedule_type
        self.max_lr = cfg.max_lr
        self.min_lr = cfg.min_lr

        self.update_count = 0

    @property
    def current_lr(self) -> float:
        # GPU to CPU sync for storing/logging learning rate.
        return self.lr_t.item()

    @current_lr.setter
    def current_lr(self, value: float) -> None:
        self.lr_t.fill_(float(value))

    def adapt_lr_device(self, kl: torch.Tensor) -> None:
        # adaptive kl, but done on GPU fully for speed.
        lr = self.lr_t
        adjusted = torch.where(
            kl > self.desired_kl * KL_LR_DECREASE_FACTOR,
            lr / LR_ADJUST_RATIO,
            torch.where(
                (kl > 0.0) & (kl < self.desired_kl / KL_LR_INCREASE_FACTOR),
                lr * LR_ADJUST_RATIO,
                lr,
            ),
        )
        self.lr_t.copy_(adjusted.clamp_(self.min_lr, self.max_lr))

    @torch.compile
    def update_normalization(self, obs: torch.Tensor) -> None:
        self.actor.update_normalization(obs)

    def save_checkpoint(self, path: str, iteration: int) -> None:
        # save a complete checkpoint for resuming training. includes weights,
        # optimizer, lr, and number of updates. allows for seamless saving of
        # checkpoints to be used for resuming later, or for experimenting with
        # fine-tuning.

        torch.save(
            {
                "iteration": iteration,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "current_lr": self.current_lr,
                "update_count": self.update_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        # fully loads the checkpoint saved by the save_checkpoint() function.

        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(strip_compile_prefix(checkpoint["actor"]))
        self.actor.log_std.data.clamp_(min=self.log_std_min, max=self.log_std_max)
        self.critic.load_state_dict(strip_compile_prefix(checkpoint["critic"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_lr = checkpoint["current_lr"]
        self.update_count = checkpoint.get("update_count", 0)
        return checkpoint["iteration"]

    def select_action(
        self, state_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # selects action based upon an observation and the current policy,
        # returns action, log_prob, mu, std.
        if not torch.is_tensor(state_obs):
            state_obs = torch.tensor(state_obs, dtype=torch.float, device=self.device)
        else:
            state_obs = state_obs.to(self.device)

        with torch.no_grad():
            return self.act(state_obs)

    @torch.compile
    def act(
        self, state_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # performs forward pass, gaussian sample, and log_prob in one compiled function
        mu, std, log_std = self.actor(state_obs)
        action = torch.addcmul(mu, std, torch.randn_like(mu))
        log_prob = gaussian_log_prob(action, mu, std, log_std)
        return action, log_prob, mu, std

    @torch.compile
    def act_deterministic(self, state_obs: torch.Tensor) -> torch.Tensor:
        # the policy mean, for evaluation/play.
        mu, _, _ = self.actor(state_obs)
        return mu

    @torch.compile
    def evaluate_values(self, states: torch.Tensor) -> torch.Tensor:
        # critic values with the trailing singleton dim dropped.
        return self.critic(states).squeeze(-1)

    def entropy(self) -> float:
        with torch.no_grad():
            return gaussian_entropy(self.actor.log_std).item()

    @torch.compile
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # computes generalized advantage estimates (GAE)

        values_extended = torch.cat([values, next_value.unsqueeze(0)], dim=0)

        deltas = (
            rewards
            + self.gamma * values_extended[1:] * (1 - dones)
            - values_extended[:-1]
        )

        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)

        for step in reversed(range(num_steps)):
            gae = deltas[step] + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae

        returns = advantages + values

        return advantages, returns

    @torch.compile(mode=COMPILE_MODE)
    def minibatch_loss(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_log_probs_old: torch.Tensor,
        batch_returns: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_values_old: torch.Tensor,
        batch_mus_old: torch.Tensor,
        std_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # single minibatch loss and kl in a single compiled function.

        # calculate log_probs for current policy.
        mu, std, log_std = self.actor(batch_states)

        log_probs = gaussian_log_prob(batch_actions, mu, std, log_std)
        entropy = gaussian_entropy(log_std)

        # full KL divergence.
        mu_d = mu.detach()
        std_d = std.detach()
        kl = (
            torch.log(std_d / (std_old + 1e-8))
            + torch.square(std_old) / (2.0 * torch.square(std_d) + 1e-8)
            - 0.5
        ).sum() + (
            torch.square(batch_mus_old - mu_d) / (2.0 * torch.square(std_d) + 1e-8)
        ).sum(dim=-1).mean()

        # compute surrogate loss
        ratios = torch.exp(log_probs - batch_log_probs_old)
        surr1 = ratios * batch_advantages
        surr2 = (
            torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * batch_advantages
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        # compute clipped value loss
        values = self.critic(batch_states).view(-1)
        value_pred_clipped = batch_values_old + torch.clamp(
            values - batch_values_old, -self.clip_epsilon, self.clip_epsilon
        )
        value_losses = (values - batch_returns).pow(2)
        value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
        critic_loss = torch.max(value_losses, value_losses_clipped).mean()

        # total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        return loss, kl

    def update(
        self,
        batch: RolloutBatch,
        epochs: int = 4,
        num_mini_batches: int = 4,
    ) -> float:
        # updates Actor and Critic networks using the PPO algorithm.

        # normalize advantages once over the full batch
        advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
        )

        dataset_size = len(batch)
        batch_size = dataset_size // num_mini_batches

        # KL accumulates on device and the adaptive schedule runs on device, so the
        # whole update needs no GPU to CPU sync until the mean is read out below.
        kl_sum = torch.zeros((), device=self.device)
        num_updates = 0

        # training loop
        for _ in range(epochs):
            # randomizes batch data
            indices = torch.randperm(dataset_size, device=self.device)

            # mini-batch updates
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                loss, kl = self.minibatch_loss(
                    batch.states[batch_indices],
                    batch.actions[batch_indices],
                    batch.log_probs_old[batch_indices],
                    batch.returns[batch_indices],
                    advantages[batch_indices],
                    batch.values_old[batch_indices],
                    batch.mus_old[batch_indices],
                    batch.std_old,
                )

                # gradient descent step, with a clipped gradient norm. called here
                # to queue calculation on gpu while the kl accumulation below is
                # enqueued.
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                kl = kl.detach()
                kl_sum += kl
                num_updates += 1
                if self.schedule_type == "adaptive":
                    self.adapt_lr_device(kl)

                # clip actor and critic norms seperately, as they can interfere.
                torch.nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.optimizer.step()

                # clamp log_std post optimizer to keep gradients useful
                self.actor.log_std.data.clamp_(
                    min=self.log_std_min, max=self.log_std_max
                )

        # average KL divergence over all minibatch updates.
        mean_kl = (kl_sum / num_updates).item()
        if math.isnan(mean_kl):
            raise RuntimeError(f"KL is NaN at update {self.update_count}")

        self.update_count += 1

        return mean_kl
