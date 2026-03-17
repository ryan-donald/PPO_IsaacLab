import argparse
from datetime import datetime
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="PPO agent evaluation for IsaacLab environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file for actor network.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the test run.")
parser.add_argument("--video_length", type=int, default=500, help="Length of recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between videos (in steps).")

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
    args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
)

env_cfg.seed = seed

# create environment
render_mode = "rgb_array" if args_cli.video else None
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

# wrap environment for video recording if requested
if args_cli.video:
    # create video directory with timestamp
    video_dir = os.path.join(
        "logs",
        "test_videos",
        args_cli.task,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(video_dir, exist_ok=True)

    # wrap with RecordVideo
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step % args_cli.video_interval == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
env.reset()


# get world positions of links using correct API
robot = env.unwrapped.scene["robot"]
body_names = robot.data.body_names
gripper_idx = body_names.index("gripper_frame_link")
base_idx = body_names.index("base_link")
gripper_pos = robot.data.body_pos_w[0, gripper_idx]
base_pos = robot.data.body_pos_w[0, base_idx]
relative_pos = gripper_pos - base_pos
print("Gripper position relative to base_link (meters):", relative_pos)

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
    entropy_coef=env_config.entropy_coef,
    use_normalization=env_config.use_normalization
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
# log_path = f"ryan_logs/{args_cli.task}/"
# checkpoint_path = log_path + "actor_best.pth"
checkpoint_path = args_cli.checkpoint
start_iteration = 0
if os.path.exists(checkpoint_path):
    print(f"\nFound existing checkpoint: {checkpoint_path}")
    agent.actor.load_state_dict(torch.load(
        checkpoint_path, map_location=device))
    print(f"Loaded checkpoint.")

print("\nStarting evaluation...\n")

# storage for episode rewards and lengths, and other plotting data
episode_rewards = []
episode_lengths = []
current_episode_rewards = torch.zeros(num_envs, device=device)
current_episode_lengths = torch.zeros(num_envs, device=device)
plot_data = []
reward_steps = []

for update in range(max_iterations):
    states = torch.zeros((num_steps, num_envs, state_dim)).to(device)
    actions = torch.zeros((num_steps, num_envs, action_dim),
                          dtype=torch.float).to(device)
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
            state_obs = state['policy'] if 'policy' in state else state[list(state.keys())[
                0]]
        else:
            state_obs = state

        # update normalization statistics
        if env_config.use_normalization:
            agent.actor.update_normalization(state_obs)
            agent.critic.update_normalization(state_obs)

        # select action from policy
        with torch.no_grad():
            mu, std = agent.actor(state_obs)

        # take step in environment
        next_state, reward, terminated, truncated, info = env.step(mu)

        state = next_state

        # accumulate episode rewards and lengths
        current_episode_rewards += rewards[step]
        current_episode_lengths += 1

        episode_done_mask = dones[step].bool()


env.close()
