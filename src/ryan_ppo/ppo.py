from __future__ import annotations

import torch
import torch.optim as optim
from torch.distributions import Normal

from ryan_ppo.network import LOG_STD_MAX, LOG_STD_MIN, Actor, Critic
from ryan_ppo.normalization import ObsNormalization


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        clip_epsilon: float = 0.2,
        hidden_dims: list[int] = [64, 64],
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.01,
        schedule_type: str = "adaptive",
        entropy_coef: float = 0.001,
        saturation_coef: float = 1e-3,
        use_normalization: bool = True,
    ) -> None:

        if use_normalization:
            self.obs_normalizer = ObsNormalization(state_dim)
        else:
            self.obs_normalizer = None

        self.device = device

        self.actor = Actor(state_dim, action_dim, hidden_dims, self.obs_normalizer).to(
            device
        )
        self.critic = Critic(state_dim, hidden_dims, self.obs_normalizer).to(device)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.optimizer = optim.Adam(self.actor_params + self.critic_params, lr=lr)

        self.actor = torch.compile(self.actor)
        self.critic = torch.compile(self.critic)

        # hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.saturation_coef = saturation_coef
        self.value_coef = value_coef
        self.desired_kl = desired_kl
        self.schedule_type = schedule_type
        self.current_lr = lr

        self.update_count = 0

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
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.log_std.data.clamp_(LOG_STD_MIN, LOG_STD_MAX)
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_lr = checkpoint["current_lr"]
        self.update_count = checkpoint.get("update_count", 0)
        return checkpoint["iteration"]

    def select_action(
        self, state_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # selects action based upon an observation and the current policy,
        # returns action, log_prob, entropy.
        if not torch.is_tensor(state_obs):
            state_obs = torch.tensor(state_obs, dtype=torch.float, device=self.device)
        else:
            state_obs = state_obs.to(self.device)

        with torch.no_grad():
            mu, std = self.actor(state_obs)

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
        # computes normalized generalized advantage estimates (GAE)

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
        batch_size: int = 64,
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

        mean_kl = 0
        num_updates = 0
        kl_abort = False

        # training loop
        for epoch in range(epochs):
            # randomizes batch data
            indices = torch.randperm(dataset_size, device=self.device)

            epoch_kl = 0

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
                mu, std = self.actor(batch_states)
                pre_tanh = self.actor.pre_tanh

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
                    epoch_kl += batch_kl
                    num_updates += 1

                    # kl early stopping per minibatch
                    if batch_kl > self.desired_kl * 4.0:
                        kl_abort = True

                    if self.schedule_type == "adaptive":
                        if batch_kl > self.desired_kl * 2.0:
                            self.current_lr = max(1e-5, self.current_lr / 1.5)
                        elif batch_kl < self.desired_kl / 2.0 and batch_kl > 0.0:
                            self.current_lr = min(1e-2, self.current_lr * 1.5)
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
                self.actor.log_std.data.clamp_(LOG_STD_MIN, LOG_STD_MAX)
            mean_kl += epoch_kl

            mean_epoch_kl = epoch_kl / num_mini_batches
            if kl_abort or mean_epoch_kl > self.desired_kl * 1.5:
                break

        # average KL divergence over all updates,
        # adjust learning rate if using adaptive schedule
        mean_kl = mean_kl / num_updates if num_updates > 0 else 0
        self.update_count += 1

        return mean_kl
