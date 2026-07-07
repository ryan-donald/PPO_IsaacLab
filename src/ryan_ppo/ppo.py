from __future__ import annotations

import math

import torch
import torch.optim as optim
from torch.distributions import Normal

from ryan_ppo.config import TrainConfig
from ryan_ppo.network import (
    GRIPPER_LOG_STD_MIN,
    LOG_STD_MAX,
    LOG_STD_MIN,
    Actor,
    Critic,
)
from ryan_ppo.normalization import ObsNormalization

KL_ABORT_FACTOR = 4.0
KL_LR_DECREASE_FACTOR = 2.0
KL_LR_INCREASE_FACTOR = 2.0
KL_EPOCH_STOP_FACTOR = 1.5
LR_ADJUST_RATIO = 1.5
MIN_LR = 1e-5
MAX_LR = 1e-3


def strip_compile_prefix(state_dict: dict) -> dict:
    # allows loading of compiled models into non-compiled models
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


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
            state_dim, action_dim, cfg.hidden_dims, self.obs_normalizer
        ).to(device)
        self.critic = Critic(state_dim, cfg.hidden_dims, self.obs_normalizer).to(device)

        # per-dim lower bound on log_std. only tasks whose last action dim is a
        # binary gripper raise that dim's floor
        self.log_std_min = torch.full((action_dim,), float(LOG_STD_MIN), device=device)
        if cfg.has_gripper_action:
            self.log_std_min[-1] = float(GRIPPER_LOG_STD_MIN)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.optimizer = optim.Adam(
            self.actor_params + self.critic_params, lr=cfg.learning_rate
        )

        # allows saving of non-compiled weights.
        self.actor_module = self.actor
        self.critic_module = self.critic

        self.actor = torch.compile(self.actor)
        self.critic = torch.compile(self.critic)

        # hyperparameters
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_epsilon = cfg.clip_epsilon
        self.max_grad_norm = cfg.max_grad_norm
        self.entropy_coef = cfg.entropy_coef
        self.saturation_coef = cfg.saturation_coef
        self.value_coef = cfg.value_coef
        self.desired_kl = cfg.desired_kl
        self.schedule_type = cfg.schedule_type
        self.current_lr = cfg.learning_rate

        self.update_count = 0

    def save_checkpoint(self, path: str, iteration: int) -> None:
        # save a complete checkpoint for resuming training. includes weights,
        # optimizer, lr, and number of updates. allows for seamless saving of
        # checkpoints to be used for resuming later, or for experimenting with
        # fine-tuning.

        torch.save(
            {
                "iteration": iteration,
                "actor": self.actor_module.state_dict(),
                "critic": self.critic_module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "current_lr": self.current_lr,
                "update_count": self.update_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        # fully loads the checkpoint saved by the save_checkpoint() function.

        checkpoint = torch.load(path, map_location=self.device)
        self.actor_module.load_state_dict(strip_compile_prefix(checkpoint["actor"]))
        self.actor.log_std.data.clamp_(max=LOG_STD_MAX)
        torch.maximum(
            self.actor.log_std.data, self.log_std_min, out=self.actor.log_std.data
        )
        self.critic_module.load_state_dict(strip_compile_prefix(checkpoint["critic"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_lr = checkpoint["current_lr"]
        self.update_count = checkpoint.get("update_count", 0)
        return checkpoint["iteration"]

    def select_action(
        self, state_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # selects action based upon an observation and the current policy,
        # returns action, log_prob, entropy, mu, std.
        if not torch.is_tensor(state_obs):
            state_obs = torch.tensor(state_obs, dtype=torch.float, device=self.device)
        else:
            state_obs = state_obs.to(self.device)

        with torch.no_grad():
            mu, std, _ = self.actor(state_obs)

        dist = Normal(mu, std)
        action = dist.sample()
        entropy = dist.entropy().sum(dim=-1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, entropy, mu, std

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

    # @torch.compile
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor,
        mus_old: torch.Tensor,
        stds_old: torch.Tensor,
        epochs: int = 4,
        num_mini_batches: int = 4,
    ) -> float:
        # updates Actor and Critic networks using the PPO algorithm

        # batch data
        b_states = states.reshape(-1, states.shape[-1])
        b_actions = actions.reshape(-1, actions.shape[-1])
        b_log_probs_old = log_probs_old.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values_old = values_old.reshape(-1)
        b_mus_old = mus_old.reshape(-1, mus_old.shape[-1])
        b_stds_old = stds_old.reshape(-1, stds_old.shape[-1])

        dataset_size = b_states.shape[0]
        batch_size = dataset_size // num_mini_batches

        mean_kl = 0
        num_updates = 0
        kl_abort = False

        # training loop
        for epoch in range(epochs):
            # randomizes batch data
            indices = torch.randperm(dataset_size, device=self.device)

            epoch_kl = 0
            epoch_updates = 0

            # mini-batch updates
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = b_states[batch_indices]
                batch_actions = b_actions[batch_indices]
                batch_log_probs_old = b_log_probs_old[batch_indices]
                batch_returns = b_returns[batch_indices]
                batch_advantages = b_advantages[batch_indices]
                batch_values_old = b_values_old[batch_indices]
                batch_mus_old = b_mus_old[batch_indices]
                batch_stds_old = b_stds_old[batch_indices]

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

                # calculate log_probs for current policy
                mu, std, pre_tanh = self.actor(batch_states)

                dist = Normal(mu, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # full KL divergence
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(std / (batch_stds_old + 1e-8))
                        + (
                            torch.square(batch_stds_old)
                            + torch.square(batch_mus_old - mu)
                        )
                        / (2.0 * torch.square(std) + 1e-8)
                        - 0.5,
                        dim=-1,
                    )
                    batch_kl = kl.mean().item()
                    if math.isnan(batch_kl):
                        raise RuntimeError(
                            f"KL is NaN at update {self.update_count}, epoch {epoch}"
                        )
                    epoch_kl += batch_kl
                    epoch_updates += 1
                    num_updates += 1

                    # kl early stopping per minibatch
                    if batch_kl > self.desired_kl * KL_ABORT_FACTOR:
                        kl_abort = True

                    if self.schedule_type == "adaptive":
                        if batch_kl > self.desired_kl * KL_LR_DECREASE_FACTOR:
                            self.current_lr = max(
                                MIN_LR, self.current_lr / LR_ADJUST_RATIO
                            )
                        elif 0.0 < batch_kl < self.desired_kl / KL_LR_INCREASE_FACTOR:
                            self.current_lr = min(
                                MAX_LR, self.current_lr * LR_ADJUST_RATIO
                            )
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.current_lr

                if kl_abort:
                    break

                # compute surrogate loss
                ratios = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
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
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # total loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                    + self.saturation_coef * pre_tanh.pow(2).mean()
                )

                # gradient descent step, with a clipped gradient norm
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.actor_params + self.critic_params,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # clamp log_std post optimizer to keep gradients useful
                self.actor.log_std.data.clamp_(max=LOG_STD_MAX)
                torch.maximum(
                    self.actor.log_std.data,
                    self.log_std_min,
                    out=self.actor.log_std.data,
                )
            mean_kl += epoch_kl

            mean_epoch_kl = epoch_kl / epoch_updates
            if kl_abort or mean_epoch_kl > self.desired_kl * KL_EPOCH_STOP_FACTOR:
                break

        # average KL divergence over all updates,
        # adjust learning rate if using adaptive schedule
        mean_kl = mean_kl / num_updates if num_updates > 0 else 0
        self.update_count += 1

        return mean_kl
