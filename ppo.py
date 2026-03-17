import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
from network import Actor, Critic


class PPOAgent:

    def __init__(self,
                 state_dim,
                 action_dim,
                 device=torch.device("cpu"),
                 lr=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 value_coef=0.5,
                 clip_epsilon=0.2,
                 hidden_dims=[64, 64],
                 max_grad_norm=1.0,
                 desired_kl=0.01,
                 schedule_type="adaptive",
                 entropy_coef=0.001,
                 use_normalization=True):

        # initialization of networks and optimizer
        self.device = device
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.critic = Critic(state_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        # self.actor = torch.compile(self.actor)
        # self.critic = torch.compile(self.critic)

        # hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.desired_kl = desired_kl
        self.schedule_type = schedule_type
        self.current_lr = lr

        self.update_count = 0

    def select_action(self, state_obs):
        # selects action based upon an observation and the current policy,
        # returns action, log_prob, entropy
        if not torch.is_tensor(state_obs):
            state_obs = torch.tensor(
                state_obs, dtype=torch.float, device=self.device)
        else:
            state_obs = state_obs.to(self.device)

        with torch.no_grad():
            mu, std = self.actor(state_obs)

        dist = Normal(mu, std)
        action = dist.sample()
        entropy = dist.entropy().sum(dim=-1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, entropy

    # @torch.compile
    def compute_gae(self, rewards, values, dones, next_value):
        # computes normalized generalized advantage estimates (GAE)

        values_extended = torch.cat([values, next_value.unsqueeze(0)], dim=0)

        deltas = rewards + self.gamma * \
            values_extended[1:] * (1-dones) - values_extended[:-1]

        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)

        for step in reversed(range(num_steps)):
            gae = deltas[step] + self.gamma * \
                self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        return advantages, returns

    # @torch.compile
    def update(self, states, actions, log_probs_old, returns, advantages, values_old, mus_old, stds_old, epochs=4, batch_size=64):
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

        mean_kl = 0.0
        num_updates = 0
        stop_early = False

        # training loop
        for epoch in range(epochs):

            # early stopping for KL divergence
            if stop_early:
                break

            # randomizes batch data
            indices = np.random.permutation(dataset_size)

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

                # calculate log_probs for current policy
                mu, std = self.actor(batch_states)

                dist = Normal(mu, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # full KL divergence
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(std / (batch_stds_old + 1e-8) + 1e-8)
                        + (torch.square(batch_stds_old) +
                           torch.square(batch_mus_old - mu))
                        / (2.0 * torch.square(std) + 1e-8)
                        - 0.5,
                        dim=-1,
                    )
                    batch_kl = kl.mean().item()
                    mean_kl += batch_kl
                    num_updates += 1

                    # early stopping: if KL has already exceeded the threshold, skip
                    # remaining mini-batches and epochs to avoid over-updating the policy
                    if batch_kl > self.desired_kl * 2.0:
                        stop_early = True
                        break

                # compute surrogate loss
                ratios = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # compute clipped value loss
                values = self.critic(batch_states).squeeze()
                value_pred_clipped = batch_values_old + torch.clamp(
                    values - batch_values_old,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_losses = (values - batch_returns).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - batch_returns).pow(2)
                critic_loss = torch.max(
                    value_losses, value_losses_clipped).mean()

                # total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # gradient descent step, with a clipped gradient norm
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) +
                    list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

        # average KL divergence over all updates, adjust learning rate if using adaptive schedule
        mean_kl = mean_kl / num_updates if num_updates > 0 else 0
        self.update_count += 1
        if self.schedule_type == "adaptive":
            if mean_kl > self.desired_kl * 2.0:
                self.current_lr = max(1e-5, self.current_lr / 1.5)
            elif mean_kl < self.desired_kl / 2.0 and mean_kl > 0.0:
                self.current_lr = min(1e-2, self.current_lr * 1.5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr

        return mean_kl
