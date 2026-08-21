from __future__ import annotations

import math
import os

import torch
import torch.optim as optim

from ryan_ppo.config import TrainConfig
from ryan_ppo.network import Actor, Critic
from ryan_ppo.normalization import ObsNormalization

KL_ABORT_FACTOR = 4.0
KL_LR_DECREASE_FACTOR = 2.0
KL_LR_INCREASE_FACTOR = 2.0
KL_EPOCH_STOP_FACTOR = 1.5
LR_ADJUST_RATIO = 1.5

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)

# performance improvements by adjusting compile mode for minibatch loss.
COMPILE_MODE = os.environ.get("RYAN_PPO_COMPILE_MODE", "max-autotune-no-cudagraphs")
if COMPILE_MODE == "default":
    COMPILE_MODE = None


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
        self.value_coef = cfg.value_coef
        self.desired_kl = cfg.desired_kl
        self.schedule_type = cfg.schedule_type
        self.max_lr = cfg.max_lr
        self.min_lr = cfg.min_lr
        self.kl_early_stop = cfg.kl_early_stop

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
        self.actor.log_std.data.clamp_(min=self.log_std_min, max=self.log_std_max)
        self.critic_module.load_state_dict(strip_compile_prefix(checkpoint["critic"]))
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
        mu, std = self.actor_module(state_obs)
        action = torch.addcmul(mu, std, torch.randn_like(mu))
        log_prob = (
            -0.5 * ((action - mu) / std).square()
            - self.actor_module.log_std
            - LOG_SQRT_2PI
        ).sum(dim=-1)
        return action, log_prob, mu, std

    def entropy(self) -> float:
        # closed-form Gaussian entropy, summed over action dims.
        with torch.no_grad():
            return (self.actor.log_std + 0.5 + LOG_SQRT_2PI).sum().item()

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
        mu, std = self.actor_module(batch_states)

        log_probs = (
            -0.5 * ((batch_actions - mu) / std).square()
            - self.actor_module.log_std
            - LOG_SQRT_2PI
        ).sum(dim=-1)
        entropy = (self.actor_module.log_std + 0.5 + LOG_SQRT_2PI).sum()

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
        values = self.critic_module(batch_states).view(-1)
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
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor,
        mus_old: torch.Tensor,
        std_old: torch.Tensor,
        epochs: int = 4,
        num_mini_batches: int = 4,
    ) -> tuple[float, float]:
        # updates Actor and Critic networks using the PPO algorithm.

        # batch data
        b_states = states.reshape(-1, states.shape[-1])
        b_actions = actions.reshape(-1, actions.shape[-1])
        b_log_probs_old = log_probs_old.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values_old = values_old.reshape(-1)
        b_mus_old = mus_old.reshape(-1, mus_old.shape[-1])

        # normalize advantages once over the full batch
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        dataset_size = b_states.shape[0]
        batch_size = dataset_size // num_mini_batches

        # accumulates KL when not using early stopping to prevent GPU to CPU syncs.
        kl_sum = torch.zeros((), device=self.device)
        mean_kl = 0.0
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

                loss, kl = self.minibatch_loss(
                    b_states[batch_indices],
                    b_actions[batch_indices],
                    b_log_probs_old[batch_indices],
                    b_returns[batch_indices],
                    b_advantages[batch_indices],
                    b_values_old[batch_indices],
                    b_mus_old[batch_indices],
                    std_old,
                )

                # gradient descent step, with a clipped gradient norm. called here
                # to queue calculation on gpu while kl calculation is performed
                # and moved gpu-cpu.
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                epoch_updates += 1
                num_updates += 1

                if self.kl_early_stop:
                    batch_kl = kl.item()
                    if math.isnan(batch_kl):
                        raise RuntimeError(
                            f"KL is NaN at update {self.update_count}, epoch {epoch}"
                        )
                    epoch_kl += batch_kl

                    # kl early stopping per minibatch
                    if batch_kl > self.desired_kl * KL_ABORT_FACTOR:
                        kl_abort = True
                        break
                else:
                    # no kl early stopping.
                    kl_sum += kl.detach()
                    if self.schedule_type == "adaptive":
                        self.adapt_lr_device(kl.detach())

                # clip actor and critic norms seperately, as they can interfere.
                torch.nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.optimizer.step()

                # clamp log_std post optimizer to keep gradients useful
                self.actor.log_std.data.clamp_(
                    min=self.log_std_min, max=self.log_std_max
                )
            mean_kl += epoch_kl

            if self.kl_early_stop:
                mean_epoch_kl = epoch_kl / epoch_updates
                if kl_abort or mean_epoch_kl > self.desired_kl * KL_EPOCH_STOP_FACTOR:
                    break

        # average KL divergence over all updates. different depending on how kl is
        # stored based on kl early stopping enabled or not.
        if self.kl_early_stop:
            mean_kl = mean_kl / num_updates if num_updates > 0 else 0
        else:
            mean_kl = (kl_sum / num_updates).item() if num_updates > 0 else 0.0
            if math.isnan(mean_kl):
                raise RuntimeError(f"KL is NaN at update {self.update_count}")

        # with kl early stopping, only update learning rate once per iteration, not per
        # minibatch.
        if self.schedule_type == "adaptive" and self.kl_early_stop:
            if mean_kl > self.desired_kl * KL_LR_DECREASE_FACTOR:
                self.current_lr = max(self.min_lr, self.current_lr / LR_ADJUST_RATIO)
            elif 0.0 < mean_kl < self.desired_kl / KL_LR_INCREASE_FACTOR:
                self.current_lr = min(self.max_lr, self.current_lr * LR_ADJUST_RATIO)

        self.update_count += 1

        # tracking how many actual epochs ran this update. i.e., how many before kl
        # stopping.
        updates_per_epoch = math.ceil(dataset_size / batch_size)
        epochs_run = num_updates / updates_per_epoch

        return mean_kl, epochs_run
