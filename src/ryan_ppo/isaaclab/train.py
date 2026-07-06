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
    from contextlib import contextmanager
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
    from ryan_ppo.utils import generate_table, get_cfg_path, policy_obs

    # set device before using it in class instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # allow TF32 tensor cores for float32 matmuls.
    torch.set_float32_matmul_precision("high")

    @contextmanager
    def profile_zone(zone_id, name):
        # wraps a block in a carb.profiler zone; no-op when profiling is off.
        if args_cli.profile:
            carb.profiler.begin(zone_id, name)
        try:
            yield
        finally:
            if args_cli.profile:
                carb.profiler.end(zone_id)

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
    num_steps = cfg.num_steps_per_env
    curr_max = -float("inf")

    # logging and checkpointing
    log_path = f"ppo_logs/{args_cli.task}/{run_id}/"
    os.makedirs(log_path, exist_ok=True)

    # iterations at which to save full resumable checkpoint bundles
    checkpoint_iters = {
        int(it) for it in args_cli.checkpoint_iters.split(",") if it.strip()
    }

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
        "Iteration": 0,
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

            # select action from policy
            with profile_zone(2, "select_action"), torch.no_grad():
                action, log_prob, entropy, mu, std = agent.select_action(state_obs)
                value = agent.critic(state_obs).squeeze(-1)

            # take step in environment
            with profile_zone(3, "env_step"):
                next_state, reward, terminated, truncated, info = env.step(action)

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

        # compute GAE advantages and returns
        with profile_zone(4, "compute_gae"):
            advantages, returns = agent.compute_gae(
                storage.rewards, storage.values, storage.dones, next_value
            )

        # update actor and critic networks
        update_start = time.perf_counter()
        with profile_zone(5, "update"):
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
                num_mini_batches=cfg.num_mini_batches,
            )
        update_time = time.perf_counter() - update_start

        with profile_zone(6, "logging/cli"):
            # reduce rollout into episode statistics (reward stats forward-filled
            # if no episodes completed this rollout; num_episodes is zero then).
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

            # aggregate rows for the TUI (added after wandb.log so they aren't
            # logged).
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
            train_stats["Iteration"] = update + 1

            live.update(
                generate_table(
                    perf_stats, train_stats, stats.term_rewards, args_cli.task, run_url
                )
            )

        # save best model when reward improves
        if args_cli.save:
            iteration = update + 1

            if stats.avg_reward > curr_max:
                curr_max = stats.avg_reward
                torch.save(agent.actor_module.state_dict(), log_path + "actor_best.pth")
                torch.save(
                    agent.critic_module.state_dict(), log_path + "critic_best.pth"
                )

            # periodic weight snapshots and a rolling resumable checkpoint
            if iteration % 100 == 0:
                torch.save(
                    agent.actor_module.state_dict(),
                    log_path + f"actor_iter_{iteration}.pth",
                )
                torch.save(
                    agent.critic_module.state_dict(),
                    log_path + f"critic_iter_{iteration}.pth",
                )
                agent.save_checkpoint(log_path + "checkpoint_latest.pth", iteration)

            # full checkpoint bundles at requested iterations, for resuming or
            # fine-tuning later
            if iteration in checkpoint_iters:
                agent.save_checkpoint(
                    log_path + f"checkpoint_{iteration}.pth", iteration
                )

    live.stop()
    env.close()
    wandb.finish()

    # save final model
    if args_cli.save:
        torch.save(agent.actor_module.state_dict(), log_path + "actor_final.pth")
        torch.save(agent.critic_module.state_dict(), log_path + "critic_final.pth")
        agent.save_checkpoint(log_path + "checkpoint_final.pth", cfg.max_iterations)

    if args_cli.profile:
        carb.profiler.end(1)

    simulation_app.close()


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
        "--checkpoint_iters",
        type=str,
        default="5000,7500,10000",
        help="Comma-separated iterations at which to save full resumable "
        "checkpoint_<iter>.pth bundles (restore weights, optimizer, LR, and "
        "iteration). Requires --save.",
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
