import argparse

from isaaclab.app import AppLauncher


def play(args_cli):
    # video recording uses rgb_array rendering, which requires cameras to be
    # enabled before the app launches (otherwise the viewport/video are blank/white)
    if args_cli.video:
        args_cli.enable_cameras = True

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import os
    import random
    from datetime import datetime

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import ryan_tasks  # noqa: F401
    from ryan_ppo.config import TrainConfig
    from ryan_ppo.ppo import PPOAgent
    from ryan_ppo.utils import get_cfg_path, policy_obs

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
    cfg = TrainConfig.from_ini(get_cfg_path(args_cli.task))

    # store state and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space["policy"].shape[1]
    else:
        state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    # initialize PPO agent
    agent = PPOAgent(state_dim, action_dim, cfg, device=device)
    agent.actor.eval()

    # reset environment
    state, info = env.reset()
    num_envs = env.unwrapped.num_envs

    print(f"Evaluating with {num_envs} environments.")

    # logging and checkpointing
    # log_path = f"ryan_logs/{args_cli.task}/"
    # checkpoint_path = log_path + "actor_best.pth"
    checkpoint_path = args_cli.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "actor" in checkpoint:
            checkpoint = checkpoint["actor"]
        agent.actor.load_state_dict(checkpoint)
        print("Loaded checkpoint.")

    print("\nStarting evaluation...\n")

    for step in range(args_cli.eval_steps):
        # handle both Dict and Box observation spaces
        state_obs = policy_obs(state)

        # deterministic (mean) action from the policy
        with torch.no_grad():
            mu, _ = agent.actor(state_obs)

        # step the environment with the deterministic action
        state, _, _, _, _ = env.step(mu)

    env.close()
    simulation_app.close()


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
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=2000,
        help="Number of steps to run the policy for.",
    )
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)

    # parse the arguments
    args_cli, _ = parser.parse_known_args()

    play(args_cli)
