import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(

    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."

)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random

import os

from ppo import PPOAgent
from env_cfgs import EnvConfig

# set device before using it in class instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set seeds for reproducibility
seed = args_cli.seed
print(f"Setting seed: {seed}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

env_cfg = parse_env_cfg(
    args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)

env_cfg.seed = seed

# create environment
env = gym.make(args_cli.task, cfg=env_cfg)
env.reset()

# get environment-specific training configuration
env_config = EnvConfig(args_cli)
num_steps_per_env = env_config.num_steps_per_env
num_mini_batches = env_config.num_mini_batches
num_learning_epochs = env_config.num_learning_epochs
max_iterations = env_config.max_iterations


# store state and action dimensions
if isinstance(env.observation_space, gym.spaces.Dict):
    state_dim = env.observation_space['policy'].shape[1]
else:
    state_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]

# initialize PPO agent
agent = PPOAgent(
    state_dim,
    action_dim,
    device=device,
    lr=env_config.lr,
    gamma=env_config.gamma,
    hidden_dims=env_config.hidden_dims,
    gae_lambda=env_config.gae_lambda,
    value_coef=env_config.value_coef,
    clip_epsilon=env_config.clip_epsilon,
    max_grad_norm=env_config.max_grad_norm,
    desired_kl=env_config.desired_kl,
    schedule_type=env_config.schedule_type,
    entropy_coef=env_config.entropy_coef
)

# reset environment
state, info = env.reset()
num_envs = env.unwrapped.num_envs

steps_per_rollout = num_steps_per_env * num_envs  # 24 * num_envs
batch_size = steps_per_rollout // num_mini_batches
num_steps = num_steps_per_env
curr_max = -float('inf')

# print training configuration
print(f"Training configuration:")
print(f"  Num environments: {num_envs}")
print(f"  Steps per env per rollout: {num_steps_per_env}")
print(f"  Total steps per rollout: {steps_per_rollout}")
print(f"  Mini-batches: {num_mini_batches}")
print(f"  Batch size: {batch_size}")
print(f"  Learning epochs: {num_learning_epochs}")
print(f"  Max iterations: {max_iterations}")
print(f"  Total timesteps: {max_iterations * steps_per_rollout:,}")


# logging and checkpointing
log_path = f"ryan_logs/{args_cli.task}/"
checkpoint_path = log_path + "actor_best.pth"
start_iteration = 0
if os.path.exists(checkpoint_path):
    print(f"\nFound existing checkpoint: {checkpoint_path}")
    response = input("Load and continue training? (y/N): ")
    if response.lower() == 'y':
        agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.critic.load_state_dict(torch.load(log_path + "critic_best.pth", map_location=device))
        print(f"Loaded checkpoint. Continuing training...")
        # Also load normalization statistics to continue from where we left off
        print("Normalization statistics preserved from checkpoint")

print("\nStarting training...\n")

# storage for episode rewards and lengths, and other plotting data
episode_rewards = []
episode_lengths = []
current_episode_rewards = torch.zeros(num_envs, device=device)
current_episode_lengths = torch.zeros(num_envs, device=device)
plot_data = []
reward_steps = []

for update in range(max_iterations):
    states = torch.zeros((num_steps, num_envs, state_dim)).to(device)
    actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float).to(device)
    log_probs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    entropies = torch.zeros((num_steps, num_envs)).to(device)
    mus = torch.zeros((num_steps, num_envs, action_dim)).to(device)
    stds = torch.zeros((num_steps, num_envs, action_dim)).to(device)

    for step in range(num_steps):
        # handle both Dict and Box observation spaces
        if isinstance(state, dict):
            state_obs = state['policy'] if 'policy' in state else state[list(state.keys())[0]]
        else:
            state_obs = state

        # update normalization statistics
        agent.actor.update_normalization(state_obs)
        agent.critic.update_normalization(state_obs)

        # select action from policy
        with torch.no_grad():
            action, log_prob, entropy = agent.select_action(state_obs)
            value = agent.critic(state_obs).squeeze()

        # take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = torch.logical_or(terminated, truncated)

        # store rollout data in tensors
        states[step] = state_obs
        actions[step] = action.to(device)
        log_probs[step] = log_prob.to(device)
        rewards[step] = reward.to(device)
        dones[step] = done.float().to(device)
        values[step] = value.to(device)
        entropies[step] = entropy.to(device)

        state = next_state

        # accumulate episode rewards and lengths
        current_episode_rewards += rewards[step]
        current_episode_lengths += 1
        
        episode_done_mask = dones[step].bool()
        
        # for any completed episodes, store their rewards and lengths
        if episode_done_mask.any():
            completed_rewards = current_episode_rewards[episode_done_mask]
            completed_lengths = current_episode_lengths[episode_done_mask]
            
            episode_rewards.extend(completed_rewards.cpu().numpy().tolist())
            episode_lengths.extend(completed_lengths.cpu().numpy().tolist())
            
            current_episode_rewards[episode_done_mask] = 0
            current_episode_lengths[episode_done_mask] = 0

    # bootstrap next value for GAE
    with torch.no_grad():
        if isinstance(state, dict):
            next_state_obs = state['policy'] if 'policy' in state else state[list(state.keys())[0]]
        else:
            next_state_obs = state

        next_value = agent.critic(next_state_obs).squeeze()
    
    # compute GAE advantages and returns
    advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

    # update actor and critic networks
    mean_kl = agent.update(states, actions, log_probs, returns, advantages, values, epochs=num_learning_epochs, batch_size=batch_size)

    # logging
    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0.0
    
    # save data for plotting (timestep, average_reward)
    current_timestep = (update + 1) * steps_per_rollout
    plot_data.append((current_timestep, avg_reward))
    
    # print progress every 10 updates
    if (update + 1) % 10 == 0:
        recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        min_reward = np.min(recent_rewards) if recent_rewards else 0
        max_reward = np.max(recent_rewards) if recent_rewards else 0
        std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0

        print(f"Update {update + 1}/{max_iterations} | "
              f"Avg: {avg_reward:.2f} ± {std_reward:.2f} | "
              f"Range: [{min_reward:.2f}, {max_reward:.2f}] | "
              f"KL: {mean_kl:.4f} | "
              f"LR: {agent.current_lr:.2e} | "
              f"Ep: {len(episode_rewards)} | "
              f"Steps: {(update + 1) * steps_per_rollout:,}")

    # save best model when reward improves
    if len(episode_rewards) >= 100 and avg_reward > curr_max:
        curr_max = avg_reward
        torch.save(agent.actor.state_dict(), log_path + "actor_best.pth")
        torch.save(agent.critic.state_dict(), log_path + "critic_best.pth")
        print(f"New best model saved with average reward: {curr_max:.2f}")
    
    # save checkpoint every 100 iterations
    if (update + 1) % 100 == 0:
        torch.save(agent.actor.state_dict(), log_path + f"actor_iter_{update+1}.pth")
        torch.save(agent.critic.state_dict(), log_path + f"critic_iter_{update+1}.pth")
        print(f"Checkpoint saved at iteration {update+1}")

    # task-specific success criteria
    if "Cartpole" in args_cli.task:
        success_threshold = 5.0
    elif "Lift" in args_cli.task:
        success_threshold = 200
    elif "Repose" in args_cli.task:
        success_threshold = 5000.0
    elif "Stack" in args_cli.task:
        success_threshold = 10.0
    elif "Drawer" in args_cli.task:
        success_threshold = 100.0
    else:
        success_threshold = 2.0  # Reach
    
    # check if task is solved
    if len(episode_rewards) >= 10000 and avg_reward >= success_threshold:
        print(f"Task solved! Average reward: {avg_reward:.2f} at update {update + 1}")
        break

env.close()

# save final model
torch.save(agent.actor.state_dict(), log_path + "actor_final.pth")
torch.save(agent.critic.state_dict(), log_path + "critic_final.pth")

# save training data for plotting
np.savez(
    log_path + "training_data.npz",
    timesteps=np.array([x[0] for x in plot_data]),
    avg_rewards=np.array([x[1] for x in plot_data]),
    episode_rewards=np.array(episode_rewards),
    episode_lengths=np.array(episode_lengths)
)
print(f"\n✓ Training data saved to {log_path}training_data.npz")

print(f"\nTraining complete! Final model saved.")
print(f"Total episodes: {len(episode_rewards)}")
if episode_rewards:
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")

reward_steps = episode_rewards
