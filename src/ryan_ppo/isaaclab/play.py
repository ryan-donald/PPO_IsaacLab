import argparse
from datetime import datetime

from isaaclab.app import AppLauncher


def play(args_cli):
    # launch omniverse app
    AppLauncher(args_cli)

    import configparser
    import os
    import random

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from ryan_ppo.ppo import PPOAgent

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
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
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

    # get environment-specific training configuration
    # env_config = EnvConfig(args_cli)
    env_config = configparser.ConfigParser()
    env_config.read(get_cfg_path(args_cli.task))

    learning_rate = float(env_config["train"]["learning_rate"])
    gamma = float(env_config["train"]["gamma"])
    num_learning_epochs = int(env_config["train"]["num_learning_epochs"])
    desired_kl = float(env_config["train"]["gamma"])
    clip_epsilon = float(env_config["train"]["gamma"])

    num_steps_per_env = int(env_config["train"]["num_steps_per_env"])
    num_mini_batches = int(env_config["train"]["num_mini_batches"])
    max_iterations = int(env_config["train"]["max_iterations"])

    # store state and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space["policy"].shape[1]
    else:
        state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    hidden_dims = env_config["policy"]["hidden_dims"]
    hidden_dims = [int(x) for x in hidden_dims.split(",")]

    # initialize PPO agent
    agent = PPOAgent(
        state_dim,
        action_dim,
        device=device,
        lr=learning_rate,
        gamma=gamma,
        hidden_dims=hidden_dims,
        gae_lambda=float(env_config["train"]["gae_lambda"]),
        value_coef=float(env_config["train"]["value_coef"]),
        clip_epsilon=clip_epsilon,
        max_grad_norm=float(env_config["train"]["max_grad_norm"]),
        desired_kl=desired_kl,
        schedule_type=env_config["train"]["schedule_type"],
        entropy_coef=float(env_config["train"]["entropy_coef"]),
    )

    # reset environment
    state, info = env.reset()
    num_envs = env.unwrapped.num_envs

    steps_per_rollout = num_steps_per_env * num_envs  # 24 * num_envs
    batch_size = steps_per_rollout // num_mini_batches
    num_steps = num_steps_per_env
    -float("inf")

    # print training configuration
    print("Training configuration:")
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
    if os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint.")

    print("\nStarting evaluation...\n")

    # storage for episode rewards and lengths, and other plotting data
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, device=device)

    for update in range(max_iterations):
        torch.zeros((num_steps, num_envs, state_dim)).to(device)
        torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float).to(device)
        torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        torch.zeros((num_steps, num_envs)).to(device)
        torch.zeros((num_steps, num_envs)).to(device)
        torch.zeros((num_steps, num_envs, action_dim)).to(device)
        torch.zeros((num_steps, num_envs, action_dim)).to(device)

        for step in range(num_steps):
            # handle both Dict and Box observation spaces
            if isinstance(state, dict):
                state_obs = (
                    state["policy"]
                    if "policy" in state
                    else state[list(state.keys())[0]]
                )
            else:
                state_obs = state

            # update normalization statistics
            if env_config["train"]["use_normalization"] == "True":
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

            dones[step].bool()

    env.close()


def get_cfg_path(task):
    from pathlib import Path

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[3]
    ini_file_path = project_root / "cfg" / f"{args_cli.task}.ini"
    if not ini_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {ini_file_path}")

    return ini_file_path


if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="PPO agent evaluation for IsaacLab environments."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint file for actor network.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of environments to simulate."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record video of the test run.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=500,
        help="Length of recorded video (in steps).",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Interval between videos (in steps).",
    )

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)

    # parse the arguments
    args_cli, _ = parser.parse_known_args()

    play(args_cli)
