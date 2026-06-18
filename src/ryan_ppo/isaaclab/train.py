import argparse

from isaaclab.app import AppLauncher


def train(args_cli):
    # launch omniverse app
    app_kwargs = {}
    if args_cli.profile:
        app_kwargs["profiler_backend"] = ["tracy"]

    app_launcher = AppLauncher(args_cli, **app_kwargs)
    simulation_app = app_launcher.app

    if args_cli.profile:
        import carb.profiler

        carb.profiler.begin(1, "train_loop")

    import os
    import random
    import signal
    import time
    from datetime import datetime

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    import wandb
    from isaaclab_tasks.utils import parse_env_cfg
    from rich.live import Live

    import ryan_tasks  # noqa: F401
    from ryan_ppo.config import TrainConfig
    from ryan_ppo.ppo import PPOAgent
    from ryan_ppo.utils import generate_table, policy_obs

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
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # get environment-specific training configuration
    cfg = TrainConfig.from_ini(get_cfg_path(args_cli.task))

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(project="PPO IsaacLab", name=f"{args_cli.task}_{run_id}")

    if args_cli.sweep:
        cfg.apply_sweep(wandb.config)

    # store state and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space["policy"].shape[1]
    else:
        state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    # initialize PPO agent
    agent = PPOAgent(
        state_dim,
        action_dim,
        device=device,
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        hidden_dims=cfg.hidden_dims,
        gae_lambda=cfg.gae_lambda,
        value_coef=cfg.value_coef,
        clip_epsilon=cfg.clip_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        desired_kl=cfg.desired_kl,
        schedule_type=cfg.schedule_type,
        entropy_coef=cfg.entropy_coef,
    )

    # reset environment
    state, info = env.reset()
    num_envs = env.unwrapped.num_envs

    steps_per_rollout = cfg.num_steps_per_env * num_envs  # 24 * num_envs
    batch_size = steps_per_rollout // cfg.num_mini_batches
    num_steps = cfg.num_steps_per_env
    curr_max = -float("inf")

    # logging and checkpointing
    log_path = f"ppo_logs/{args_cli.task}/{run_id}/"
    os.makedirs(log_path, exist_ok=True)

    # resume from a checkpoint. fully resumes the training, with actor, critic,
    # optimizer, lr, and step count for correct curriculum firing.
    start_iter = 0
    if args_cli.resume:
        start_iter = agent.load_checkpoint(args_cli.resume)
        env.unwrapped.common_step_counter = start_iter * cfg.num_steps_per_env
        print(f"Resumed from {args_cli.resume} at iteration {start_iter}.")

    # storage for episode rewards and lengths, and other plotting data
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, device=device)

    # per-term reward logging
    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms
    num_terms = len(term_names)
    current_term_rewards = torch.zeros(num_envs, num_terms, device=device)

    # track last known metrics for forward-filling when
    # rollouts have zero completed episodes
    num_episodes_completed = 0
    last_avg_reward = 0.0
    last_min_reward = 0.0
    last_max_reward = 0.0
    last_std_reward = 0.0
    last_avg_entropy = 0.0
    last_term_rewards = {name: 0.0 for name in term_names}

    perf_stats = {
        "steps": 0,
        "steps/s": 0.0,
        "Rollout Time": 0.0,
        "Update Time": 0.0,
        "episodes": 0.0,
        "Runtime": 0.0,
        "Remaining Time": 0.0,
    }

    train_stats = {
        "lr": 0.0,
        "kl": 0.0,
        "Epochs": 0.0,
    }

    run_url = run.url

    live = Live(
        generate_table(
            perf_stats, train_stats, last_term_rewards, args_cli.task, run_url
        ),
        refresh_per_second=4,
    )
    live.start()

    # with ctrl+c closing the script, the terminal breaks without this.
    def _on_sigint(signum, frame):
        live.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    start_time = time.perf_counter()

    # allocate tensors for storing rollout info
    states = torch.zeros((num_steps, num_envs, state_dim)).to(device)
    actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float).to(
        device
    )
    log_probs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    entropies = torch.zeros((num_steps, num_envs)).to(device)
    mus = torch.zeros((num_steps, num_envs, action_dim)).to(device)
    stds = torch.zeros((num_steps, num_envs, action_dim)).to(device)

    # preallocate history buffers for calculating episode metrics outside the step loop
    historic_episode_rewards = torch.zeros((num_steps, num_envs), device=device)
    historic_episode_lengths = torch.zeros((num_steps, num_envs), device=device)
    historic_term_rewards = torch.zeros((num_steps, num_envs, num_terms), device=device)

    for update in range(start_iter, cfg.max_iterations):
        rollout_start = time.perf_counter()
        for step in range(num_steps):
            # handle both Dict and Box observation spaces
            state_obs = policy_obs(state)

            # update normalization statistics
            if cfg.use_normalization:
                agent.actor.update_normalization(state_obs)

            if args_cli.profile:
                carb.profiler.begin(2, "select_action")

            # select action from policy
            with torch.no_grad():
                action, log_prob, entropy, mu, std = agent.select_action(state_obs)
                value = agent.critic(state_obs).squeeze(-1)

            if args_cli.profile:
                carb.profiler.end(2)

            if args_cli.profile:
                carb.profiler.begin(3, "env_step")
            # take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)

            if args_cli.profile:
                carb.profiler.end(3)
            done = torch.logical_or(terminated, truncated)

            reward = reward + agent.gamma * value * truncated.float()

            # store rollout data in tensors
            states[step] = state_obs
            actions[step] = action.to(device)
            log_probs[step] = log_prob.to(device)
            rewards[step] = reward.to(device)
            dones[step] = done.float().to(device)
            values[step] = value.to(device)
            entropies[step] = entropy.to(device)
            mus[step] = mu.to(device)
            stds[step] = std.to(device)

            state = next_state

            # accumulate episode rewards and lengths
            current_episode_rewards += rewards[step]
            current_episode_lengths += 1

            # accumulate per-term rewards: _step_reward shape is (num_envs, num_terms)
            current_term_rewards += reward_manager._step_reward.detach()

            # stores current episode data for finished episodes
            historic_episode_rewards[step] = current_episode_rewards
            historic_episode_lengths[step] = current_episode_lengths
            historic_term_rewards[step] = current_term_rewards

            # resets the current episode data tracking for finished episodes
            not_done_mask = 1.0 - dones[step]
            current_episode_rewards *= not_done_mask
            current_episode_lengths *= not_done_mask
            current_term_rewards *= not_done_mask.unsqueeze(-1)

        rollout_time = time.perf_counter() - rollout_start

        # bootstrap next value for GAE
        with torch.no_grad():
            next_value = agent.critic(policy_obs(state)).squeeze(-1)

        if args_cli.profile:
            carb.profiler.begin(4, "compute_gae")

        # compute GAE advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        if args_cli.profile:
            carb.profiler.end(4)

        if args_cli.profile:
            carb.profiler.begin(5, "update")

        # update actor and critic networks
        update_start = time.perf_counter()
        mean_kl = agent.update(
            states,
            actions,
            log_probs,
            returns,
            advantages,
            values,
            mus,
            stds,
            epochs=cfg.num_learning_epochs,
            batch_size=batch_size,
            num_mini_batches=cfg.num_mini_batches,
        )
        update_time = time.perf_counter() - update_start

        if args_cli.profile:
            carb.profiler.end(5)

        if args_cli.profile:
            carb.profiler.begin(6, "logging/cli")

        # logging, minimize GPU -> CPU transfer to improve performance
        done_mask = dones.bool()
        num_completed = done_mask.sum().item()

        if num_completed > 0:
            # returns only values for finished episodes
            completed_rewards = historic_episode_rewards[done_mask]
            num_episodes_completed = completed_rewards.numel()

            # calculates stats in pytorch on GPU, then move single value to CPU
            avg_reward = completed_rewards.mean().item()
            min_reward = completed_rewards.min().item()
            max_reward = completed_rewards.max().item()
            std_reward = completed_rewards.std().item() if num_completed > 1 else 0.0

            avg_entropy = entropies.mean().item()

            # update trackers
            last_avg_reward = avg_reward
            last_min_reward = min_reward
            last_max_reward = max_reward
            last_std_reward = std_reward

            last_avg_entropy = avg_entropy

            for t_idx, t_name in enumerate(term_names):
                completed_terms = historic_term_rewards[:, :, t_idx][done_mask]
                avg_term = completed_terms.mean().item()
                last_term_rewards[t_name] = avg_term

        else:
            # forward-fill if no new episodes finished
            avg_reward = last_avg_reward
            min_reward = last_min_reward
            max_reward = last_max_reward
            std_reward = last_std_reward

            avg_entropy = last_avg_entropy

        logging_dict = {
            "train/avg_reward": avg_reward,
            "train/min_reward": min_reward,
            "train/max_reward": max_reward,
            "train/std_reward": std_reward,
            "train/kl": mean_kl,
            "train/lr": agent.current_lr,
            "train/episodes": num_episodes_completed,
            "train/avg_entropy": avg_entropy,
        }

        for t_name in term_names:
            logging_dict[f"rewards/{t_name}"] = last_term_rewards[t_name]

        wandb.log(logging_dict, step=update)

        last_term_rewards["Mean Reward"] = avg_reward
        last_term_rewards["Max Reward"] = max_reward

        perf_stats["steps"] += steps_per_rollout
        perf_stats["Runtime"] = time.perf_counter() - start_time
        perf_stats["steps/s"] = perf_stats["steps"] / perf_stats["Runtime"]
        perf_stats["Rollout Time"] = rollout_time
        perf_stats["Update Time"] = update_time
        perf_stats["episodes"] += num_episodes_completed
        perf_stats["Remaining Time"] = (
            (cfg.max_iterations - (update + 1))
            * steps_per_rollout
            / perf_stats["steps/s"]
        )

        train_stats["lr"] = agent.current_lr
        train_stats["kl"] = mean_kl
        train_stats["Epochs"] = update + 1

        live.update(
            generate_table(
                perf_stats, train_stats, last_term_rewards, args_cli.task, run_url
            )
        )

        if args_cli.profile:
            carb.profiler.end(6)

        # save best model when reward improves
        if args_cli.save:
            if avg_reward > curr_max:
                curr_max = avg_reward
                torch.save(agent.actor.state_dict(), log_path + "actor_best.pth")
                torch.save(agent.critic.state_dict(), log_path + "critic_best.pth")

            # save checkpoint every 100 iterations
            if (update + 1) % 100 == 0:
                torch.save(
                    agent.actor.state_dict(), log_path + f"actor_iter_{update + 1}.pth"
                )
                torch.save(
                    agent.critic.state_dict(),
                    log_path + f"critic_iter_{update + 1}.pth",
                )

            # full checkpoint save during training for resuming to finetune later
            if (update + 1) == args_cli.checkpoint_iter:
                agent.save_checkpoint(log_path + "checkpoint_pretrain.pth", update + 1)

            if (update + 1) % 100 == 0:
                agent.save_checkpoint(log_path + "checkpoint_latest.pth", update + 1)

    live.stop()
    env.close()
    wandb.finish()

    # save final model
    if args_cli.save:
        torch.save(agent.actor.state_dict(), log_path + "actor_final.pth")
        torch.save(agent.critic.state_dict(), log_path + "critic_final.pth")
        agent.save_checkpoint(log_path + "checkpoint_final.pth", cfg.max_iterations)

    if args_cli.profile:
        carb.profiler.end(1)

    simulation_app.close()


def get_cfg_path(task):
    from pathlib import Path

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[3]
    ini_file_path = project_root / "cfg" / f"{task}.ini"
    if not ini_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {ini_file_path}")

    return ini_file_path


if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="Random agent for Isaac Lab environments."
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of environments to simulate."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Enable WandB parameter sweeping."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Enable saving of agents (Policy and Value networks)"
        "every 100 updates, new best performance, and at the end",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint_*.pth bundle to resume training from "
        "(restores weights, optimizer, LR, and iteration).",
    )
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        default=5000,
        help="Iteration at which to save the single resumable checkpoint_pretrain.pth "
        "bundle (default 5000, the curriculum-fire point). Requires --save.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the training loop using cProfile.",
    )

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)

    # parse the arguments
    args_cli, _ = parser.parse_known_args()

    import sys

    if args_cli.profile:
        print("Profiling enabled using carb.profiler with Tracy backend.")
        sys.argv.extend(
            [
                "--enable",
                "omni.kit.profiler.tracy",
                "--/profiler/enabled=true",
                "--/app/profilerBackend=tracy",
                "--/privacy/externalBuild=0",
                "--/app/profileFromStart=true",
            ]
        )

    train(args_cli)
