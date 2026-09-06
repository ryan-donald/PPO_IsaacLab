import argparse
import sys

import typing_extensions  # noqa: F401 # for profiling so torch doesnt break
from isaaclab.app import AppLauncher


def train(args_cli):
    # launch omniverse app
    app_kwargs = {}
    if args_cli.profile:
        app_kwargs["profiler_backend"] = ["tracy"]

    app_launcher = AppLauncher(args_cli, **app_kwargs)
    simulation_app = app_launcher.app

    import signal
    from datetime import datetime

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    import wandb
    from isaaclab_tasks.utils import parse_env_cfg

    import ryan_tasks  # noqa: F401
    from ryan_ppo.checkpointing import CheckpointSaver
    from ryan_ppo.config import TrainConfig
    from ryan_ppo.ppo import PPOAgent
    from ryan_ppo.storage import RolloutStorage
    from ryan_ppo.tracking import EpisodeTracker, TrainingLogger
    from ryan_ppo.utils import (
        PhaseTimer,
        Profiler,
        env_dims,
        get_cfg_path,
        get_device,
        install_rich_traceback,
        policy_obs,
        set_seed,
    )

    profiler = Profiler(args_cli.profile)
    profiler.begin(1, "train_loop")

    # the suppressed-library list lives in utils.install_rich_traceback.
    rich_excepthook = install_rich_traceback()

    torch.set_float32_matmul_precision("high")

    set_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.seed = args_cli.seed

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = get_device(env.unwrapped.device)

    # get environment-specific training configuration
    cfg = TrainConfig.from_ini(get_cfg_path(args_cli.task))

    if args_cli.max_iterations is not None:
        cfg.max_iterations = args_cli.max_iterations

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(
        project="PPO IsaacLab",
        name=f"{args_cli.task}_{run_id}",
        settings={"console": "off"},
    )

    # wandb.init() replaces sys.excepthook; put rich's back or a failure prints
    # the plain traceback first and the rich one second.
    sys.excepthook = rich_excepthook

    if args_cli.sweep:
        cfg.apply_sweep(wandb.config)

    state_dim, action_dim = env_dims(env)

    # initialize PPO agent
    agent = PPOAgent(state_dim, action_dim, cfg, device=device)

    # reset environment
    state, info = env.reset()
    num_envs = env.unwrapped.num_envs

    # initializes each env to a random step number in [0, episode_length].
    if cfg.stagger_initial_episodes:
        ep_len_buf = env.unwrapped.episode_length_buf
        env.unwrapped.episode_length_buf = torch.randint_like(
            ep_len_buf, high=int(env.unwrapped.max_episode_length)
        )

    steps_per_rollout = cfg.num_steps_per_env * num_envs  # 24 * num_envs
    num_steps = cfg.num_steps_per_env

    saver = CheckpointSaver(
        f"ppo_logs/{args_cli.task}/{run_id}/", args_cli.checkpoint_iters
    )

    # resume from a checkpoint. fully resumes the training, with actor, critic,
    # optimizer, lr, and step count for correct curriculum firing.
    start_iter = 0
    if args_cli.resume:
        start_iter = agent.load_checkpoint(args_cli.resume)
        env.unwrapped.common_step_counter = start_iter * cfg.num_steps_per_env
        print(f"Resumed from {args_cli.resume} at iteration {start_iter}.")
    elif cfg.use_normalization:
        # ensure that statistics are valid for first rollout on resume.
        agent.update_normalization(policy_obs(state))

    # per-term reward logging
    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms
    step_dt = env.unwrapped.step_dt

    # tracks episode/term rewards across each rollout, forward-filling reward stats
    # when a rollout has zero completed episodes.
    tracker = EpisodeTracker(num_envs, term_names, device)

    # owns the wandb stream and the live terminal table.
    logger = TrainingLogger(
        args_cli.task, run.url, term_names, cfg.max_iterations, steps_per_rollout
    )
    logger.start()

    # with ctrl+c closing the script, the terminal breaks without this.
    def _on_sigint(signum, frame):
        logger.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    # rollout buffers for one PPO iteration.
    storage = RolloutStorage(
        num_steps, num_envs, state_dim, action_dim, len(term_names), device
    )
    rollout_timer = PhaseTimer(device)
    preparation_timer = PhaseTimer(device)
    update_timer = PhaseTimer(device)

    for update in range(start_iter, cfg.max_iterations):
        rollout_timer.start()
        for step in range(num_steps):
            # handle both Dict and Box observation spaces
            state_obs = policy_obs(state)

            # select action from policy
            with profiler.zone(2, "select_action"), torch.no_grad():
                action, log_prob, mu, std = agent.select_action(state_obs)

            # take step in environment
            with profiler.zone(3, "env_step"):
                next_state, reward, terminated, truncated, info = env.step(action)

            # store steps where envs finished, either terminated or truncated
            done = torch.logical_or(terminated, truncated)

            # store rollout data.
            storage.add(
                step,
                state=state_obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done.float(),
                trunc=truncated.float(),
                term_reward=reward_manager._step_reward.detach() * step_dt,
                mu=mu,
            )

            # update state for next step
            state = next_state

        rollout_timer.stop()
        preparation_timer.start()

        # critic values for the whole rollout in one batched forward pass.
        with torch.no_grad():
            values = agent.evaluate_values(storage.states)
            next_value = agent.evaluate_values(policy_obs(state))

        # accumulate episode/term rewards
        tracker.record_rollout(storage.rewards, storage.dones, storage.term_rewards)

        # bootstrap the reward for steps where env truncated
        storage.rewards += agent.gamma * values * storage.truncs

        # compute GAE advantages and returns
        with profiler.zone(4, "compute_gae"):
            advantages, returns = agent.compute_gae(
                storage.rewards, values, storage.dones, next_value
            )

        # update actor and critic networks
        preparation_timer.stop()
        update_timer.start()
        with profiler.zone(5, "update"):
            batch = storage.flatten(
                returns=returns,
                advantages=advantages,
                values_old=values,
                std_old=std,
            )
            mean_kl = agent.update(
                batch,
                epochs=cfg.num_learning_epochs,
                num_mini_batches=cfg.num_mini_batches,
            )
        # ensure normalization statistics are fixed for each rollout + update.
        if cfg.use_normalization:
            agent.update_normalization(storage.states.view(-1, state_dim))
        update_timer.stop()

        with profiler.zone(6, "logging/cli"):
            # reduce rollout into episode statistics (reward stats forward-filled
            # if no episodes completed this rollout; num_episodes is zero then).
            stats = tracker.summarize(agent.entropy())

            logger.log_iteration(
                update,
                stats,
                mean_kl,
                agent.current_lr,
                rollout_timer.seconds(),
                update_timer.seconds(),
                preparation_time=preparation_timer.seconds(),
                warmup=(update == start_iter),
            )

        if args_cli.save:
            saver.save_iteration(
                agent, update + 1, stats.avg_reward, num_episodes=stats.num_episodes
            )

    logger.stop()
    env.close()
    wandb.finish()

    if args_cli.save:
        saver.save_final(agent, cfg.max_iterations)

    profiler.end(1)

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
        "--max_iterations",
        type=int,
        default=None,
        help="Override the config file's max_iterations.",
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

    try:
        train(args_cli)
    except KeyboardInterrupt:
        import wandb

        wandb.finish(exit_code=255)
        sys.exit(130)
    except BaseException:
        import wandb

        wandb.finish(exit_code=1)
        raise
