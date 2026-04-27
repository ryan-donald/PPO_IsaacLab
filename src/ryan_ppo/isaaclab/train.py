import argparse
from isaaclab.app import AppLauncher

def train(args_cli):
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import isaaclab_tasks
    import ryan_tasks
    from isaaclab_tasks.utils import parse_env_cfg

    import torch
    import numpy as np
    import random
    from datetime import datetime
    import configparser
    import os
    import wandb

    from ryan_ppo.ppo import PPOAgent
    from ryan_ppo.env_cfgs import EnvConfig

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
    # env_config = EnvConfig(args_cli)
    env_config = configparser.ConfigParser()
    env_config.read(get_cfg_path(args_cli.task))

    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wandb.init(project="PPO IsaacLab", name=f'{args_cli.task}_{run_id}')

    learning_rate = float(env_config['train']['learning_rate'])
    gamma = float(env_config['train']['gamma'])
    num_learning_epochs = int(env_config['train']['num_learning_epochs'])
    desired_kl = float(env_config['train']['desired_kl'])
    clip_epsilon = float(env_config['train']['clip_epsilon'])

    if args_cli.sweep:

        if 'lr' in wandb.config:
            learning_rate = wandb.config.lr
        if 'entropy_coef' in wandb.config:
            entropy_coef = wandb.config.entropy_coef
        if 'gamma' in wandb.config:
            gamma = wandb.config.gamma
        if 'num_learning_epochs' in wandb.config:
            num_learning_epochs = wandb.config.num_learning_epochs
        if 'desired_kl' in wandb.config:
            desired_kl = wandb.config.desired_kl
        if 'clip_epsilon' in wandb.config:
            clip_epsilon = wandb.config.clip_epsilon

    num_steps_per_env = int(env_config['train']['num_steps_per_env'])
    num_mini_batches = int(env_config['train']['num_mini_batches'])
    max_iterations = int(env_config['train']['max_iterations'])


    # store state and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space['policy'].shape[1]
    else:
        state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    hidden_dims = env_config['policy']['hidden_dims']
    hidden_dims = [int(x) for x in hidden_dims.split(',')]

    # initialize PPO agent
    agent = PPOAgent(
        state_dim,
        action_dim,
        device=device,
        lr=learning_rate,
        gamma=gamma,
        hidden_dims=hidden_dims,
        gae_lambda=float(env_config['train']['gae_lambda']),
        value_coef=float(env_config['train']['value_coef']),
        clip_epsilon=clip_epsilon,
        max_grad_norm=float(env_config['train']['max_grad_norm']),
        desired_kl=desired_kl,
        schedule_type=env_config['train']['schedule_type'],
        entropy_coef=float(env_config['train']['entropy_coef'])
    )

    # reset environment
    state, info = env.reset()
    num_envs = env.unwrapped.num_envs

    steps_per_rollout = num_steps_per_env * num_envs  # 24 * num_envs
    batch_size = steps_per_rollout // num_mini_batches
    num_steps = num_steps_per_env
    curr_max = -float('inf')

    # print training configuration
    # print(f"Training configuration:")
    # print(f"  Num environments: {num_envs}")
    # print(f"  Steps per env per rollout: {num_steps_per_env}")
    # print(f"  Total steps per rollout: {steps_per_rollout}")
    # print(f"  Mini-batches: {num_mini_batches}")
    # print(f"  Batch size: {batch_size}")
    # print(f"  Learning epochs: {num_learning_epochs}")
    # print(f"  Max iterations: {max_iterations}")
    # print(f"  Total timesteps: {max_iterations * steps_per_rollout:,}")


    # logging and checkpointing
    log_path = f"ppo_logs/{args_cli.task}/{run_id}/"
    os.makedirs(log_path, exist_ok=True)
    checkpoint_path = log_path + "actor_final.pth"

    start_iteration = 0
    # if os.path.exists(checkpoint_path):
    #     print(f"\nFound existing checkpoint: {checkpoint_path}")
    #     response = input("Load and continue training? (y/N): ")
    #     if response.lower() == 'y':
    #         agent.actor.load_state_dict(torch.load(
    #             checkpoint_path, map_location=device))
    #         agent.critic.load_state_dict(torch.load(
    #             log_path + "critic_final.pth", map_location=device))
    #         print(f"Loaded checkpoint. Continuing training...")

    # print("\nStarting training...\n")

    # storage for episode rewards and lengths, and other plotting data
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, device=device)
    plot_data = []
    reward_steps = []

    # per-term reward logging
    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms
    num_terms = len(term_names)
    current_term_rewards = torch.zeros(
        num_envs, num_terms, device=device)  # accumulator per env
    episode_term_rewards = {name: []
                            for name in term_names}   # completed episode sums
    # print(f"Logging {num_terms} reward terms: {term_names}")

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
            if env_config['train']['use_normalization'] == "True":
                agent.actor.update_normalization(state_obs)
                agent.critic.update_normalization(state_obs)

            # select action from policy
            with torch.no_grad():
                action, log_prob, entropy = agent.select_action(state_obs)
                value = agent.critic(state_obs).squeeze()

                # Get mu and std for KL divergence computation
                mu, std = agent.actor(state_obs)

            # take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
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

            episode_done_mask = dones[step].bool()

            # for any completed episodes, store their rewards and lengths
            if episode_done_mask.any():
                completed_rewards = current_episode_rewards[episode_done_mask]
                completed_lengths = current_episode_lengths[episode_done_mask]

                episode_rewards.extend(completed_rewards.cpu().numpy().tolist())
                episode_lengths.extend(completed_lengths.cpu().numpy().tolist())

                current_episode_rewards[episode_done_mask] = 0
                current_episode_lengths[episode_done_mask] = 0

                # store per-term episode sums for completed envs
                for t_idx, t_name in enumerate(term_names):
                    completed_term = current_term_rewards[episode_done_mask, t_idx]
                    episode_term_rewards[t_name].extend(
                        completed_term.cpu().numpy().tolist())
                current_term_rewards[episode_done_mask] = 0

        # bootstrap next value for GAE
        with torch.no_grad():
            if isinstance(state, dict):
                next_state_obs = state['policy'] if 'policy' in state else state[list(state.keys())[
                    0]]
            else:
                next_state_obs = state

            next_value = agent.critic(next_state_obs).squeeze()

        # compute GAE advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # update actor and critic networks
        mean_kl = agent.update(states, actions, log_probs, returns, advantages,
                            values, mus, stds, epochs=num_learning_epochs, batch_size=batch_size)

        # logging
        avg_reward = np.mean(episode_rewards[-100:]) if len(
            episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0.0

        # save data for plotting (timestep, average_reward)
        current_timestep = (update + 1) * steps_per_rollout
        plot_data.append((current_timestep, avg_reward))

        # logging every update — tensorboard, print every 10
        recent_rewards = episode_rewards[-100:] if len(
            episode_rewards) >= 100 else episode_rewards
        min_reward = np.min(recent_rewards) if recent_rewards else 0
        max_reward = np.max(recent_rewards) if recent_rewards else 0
        std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0

        wandb.log({"train/avg_reward": avg_reward})
        wandb.log({"train/min_reward": min_reward})
        wandb.log({"train/max_reward": max_reward})
        wandb.log({"train/std_reward": std_reward})
        wandb.log({"train/kl": mean_kl})
        wandb.log({"train/lr": agent.current_lr})
        wandb.log({"train/episodes": len(episode_rewards)})

        for t_name in term_names:
            recent_term = episode_term_rewards[t_name][-100:] if len(
                episode_term_rewards[t_name]) >= 100 else episode_term_rewards[t_name]
            avg_term = np.mean(recent_term) if recent_term else 0.0
            wandb.log({f"rewards/{t_name}": avg_term})

        # if (update + 1) % 10 == 0:
        #     print(f"Update {update + 1}/{max_iterations} | "
        #         f"Avg: {avg_reward:.2f} ± {std_reward:.2f} | "
        #         f"Range: [{min_reward:.2f}, {max_reward:.2f}] | "
        #         f"KL: {mean_kl:.4f} | "
        #         f"LR: {agent.current_lr:.2e} | "
        #         f"Ep: {len(episode_rewards)} | "
        #         f"Steps: {(update + 1) * steps_per_rollout:,}")

        # save best model when reward improves
        if (args_cli.save):
            if len(episode_rewards) >= 100 and avg_reward > curr_max:
                curr_max = avg_reward
                torch.save(agent.actor.state_dict(), log_path + "actor_best.pth")
                torch.save(agent.critic.state_dict(), log_path + "critic_best.pth")
                print(f"New best model saved with average reward: {curr_max:.2f}")

            # save checkpoint every 100 iterations
            if (update + 1) % 100 == 0:
                torch.save(agent.actor.state_dict(), log_path +
                        f"actor_iter_{update+1}.pth")
                torch.save(agent.critic.state_dict(), log_path +
                        f"critic_iter_{update+1}.pth")
                print(f"Checkpoint saved at iteration {update+1}")

    env.close()
    wandb.finish()
    simulation_app.close()

    # save final model
    if (args_cli.save):
        torch.save(agent.actor.state_dict(), log_path + "actor_final.pth")
        torch.save(agent.critic.state_dict(), log_path + "critic_final.pth")

def get_cfg_path(task):
    from pathlib import Path
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[3] 
    ini_file_path = project_root / 'cfg' / f'{args_cli.task}.ini'
    if not ini_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {ini_file_path}")
    
    return ini_file_path

if __name__ == "__main__":

    # add argparse arguments
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--sweep", action="store_true", help="Enable WandB parameter sweeping.")
    parser.add_argument("--save", action="store_true", help="Enable saving of agents (Policy and Value networks)" \
                                                            "every 100 updates, new best performance, and at the end")

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)

    # parse the arguments
    args_cli, _ = parser.parse_known_args()

    train(args_cli)
