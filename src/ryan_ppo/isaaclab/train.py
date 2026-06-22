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
    from ryan_ppo.storage import RolloutStorage
    from ryan_ppo.tracking import EpisodeTracker
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
    run = wandb.init(
        project="PPO IsaacLab",
        name=f"{args_cli.task}_{run_id}",
        settings={"console": "off"},
    )

    if args_cli.sweep:
        cfg.apply_sweep(wandb.config)

    # store state and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space["policy"].shape[1]
    else:
        state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    # initialize PPO agent
    agent = PPOAgent(state_dim, action_dim, cfg, device=device)

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

    # per-term reward logging
    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms

    # tracks episode/term rewards across each rollout, forward-filling reward stats
    # when a rollout has zero completed episodes.
    tracker = EpisodeTracker(num_envs, term_names, device)

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
            perf_stats,
            train_stats,
            {name: 0.0 for name in term_names},
            args_cli.task,
            run_url,
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

    # rollout buffers for one PPO iteration.
    storage = RolloutStorage(num_steps, num_envs, state_dim, action_dim, device)

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

            # store steps where envs finished, either terminated or truncated
            done = torch.logical_or(terminated, truncated)

            # bootstrap the reward for steps where env terminated, without this the
            # agent sees a lower reward than if non-terminal, and can avoid a state
            # more than it should
            reward = reward + agent.gamma * value * truncated.float()

            # store done values as floats, used in GAE computation later
            done_f = done.float()

            # store rollout data
            storage.add(
                step,
                state=state_obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done_f,
                value=value,
                entropy=entropy,
                mu=mu,
                std=std,
            )

            # update state for next step
            state = next_state

            # accumulate episode/term rewards (per-term _step_reward is
            # (num_envs, num_terms)).
            tracker.record_step(reward, done_f, reward_manager._step_reward.detach())

        rollout_time = time.perf_counter() - rollout_start

        # bootstrap next value for GAE
        with torch.no_grad():
            next_value = agent.critic(policy_obs(state)).squeeze(-1)

        if args_cli.profile:
            carb.profiler.begin(4, "compute_gae")

        # compute GAE advantages and returns
        advantages, returns = agent.compute_gae(
            storage.rewards, storage.values, storage.dones, next_value
        )

        if args_cli.profile:
            carb.profiler.end(4)

        if args_cli.profile:
            carb.profiler.begin(5, "update")

        # update actor and critic networks
        update_start = time.perf_counter()
        mean_kl = agent.update(
            storage.states,
            storage.actions,
            storage.log_probs,
            returns,
            advantages,
            storage.values,
            storage.mus,
            storage.stds,
            epochs=cfg.num_learning_epochs,
            batch_size=batch_size,
            num_mini_batches=cfg.num_mini_batches,
        )
        update_time = time.perf_counter() - update_start

        if args_cli.profile:
            carb.profiler.end(5)

        if args_cli.profile:
            carb.profiler.begin(6, "logging/cli")

        # reduce rollout into episode statistics (reward stats forward-filled if no
        # episodes completed this rollout; num_episodes is zero in that case).
        stats = tracker.summarize(storage.entropies)

        logging_dict = {
            "train/avg_reward": stats.avg_reward,
            "train/min_reward": stats.min_reward,
            "train/max_reward": stats.max_reward,
            "train/std_reward": stats.std_reward,
            "train/kl": mean_kl,
            "train/lr": agent.current_lr,
            "train/episodes": stats.num_episodes,
            "train/avg_entropy": stats.avg_entropy,
        }

        for t_name in term_names:
            logging_dict[f"rewards/{t_name}"] = stats.term_rewards[t_name]

        wandb.log(logging_dict, step=update)

        # aggregate rows for the TUI (added after wandb.log so they aren't logged).
        stats.term_rewards["Mean Reward"] = stats.avg_reward
        stats.term_rewards["Max Reward"] = stats.max_reward

        perf_stats["steps"] += steps_per_rollout
        perf_stats["Runtime"] = time.perf_counter() - start_time
        perf_stats["steps/s"] = perf_stats["steps"] / perf_stats["Runtime"]
        perf_stats["Rollout Time"] = rollout_time
        perf_stats["Update Time"] = update_time
        perf_stats["episodes"] += stats.num_episodes
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
                perf_stats, train_stats, stats.term_rewards, args_cli.task, run_url
            )
        )

        if args_cli.profile:
            carb.profiler.end(6)

        # save best model when reward improves
        if args_cli.save:
            if stats.avg_reward > curr_max:
                curr_max = stats.avg_reward
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

            if (update + 1) == 5000:
                agent.save_checkpoint(log_path + "checkpoint_5000.pth", update + 1)

            if (update + 1) == 7500:
                agent.save_checkpoint(log_path + "checkpoint_7500.pth", update + 1)

            if (update + 1) == 10000:
                agent.save_checkpoint(log_path + "checkpoint_10000.pth", update + 1)

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
