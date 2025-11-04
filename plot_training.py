#!/usr/bin/env python3
"""
Simple script to plot training results from saved training data.

Usage:
    python plot_training.py ryan_logs/Isaac-Reach-Franka-v0/training_data.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

def smooth_curve(data, window_size=10):
    """
    Smooth data using a moving average.
    
    Args:
        data: Array to smooth
        window_size: Size of smoothing window
    
    Returns:
        Smoothed array
    """
    if window_size <= 1:
        return data
    return uniform_filter1d(data, size=window_size, mode='nearest')

def compute_episode_timesteps(episode_lengths):
    """
    Convert episode lengths to cumulative timesteps.
    
    Args:
        episode_lengths: Array of episode lengths
    
    Returns:
        Array of cumulative timesteps at episode end
    """
    return np.cumsum(episode_lengths)

def compute_rolling_average(episode_rewards, window=100):
    """
    Compute rolling average of episode rewards.
    
    Args:
        episode_rewards: Array of episode rewards
        window: Window size for rolling average
    
    Returns:
        Array of rolling averages
    """
    rolling_avg = np.zeros(len(episode_rewards))
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - window + 1)
        rolling_avg[i] = np.mean(episode_rewards[start_idx:i+1])
    return rolling_avg

def plot_training_data(data_path, save_path=None, smooth_window=10, show_raw=False, use_episodes=False):
    """Plot average reward vs timesteps from saved training data."""
    
    # Load data
    data = np.load(data_path)
    episode_rewards = data['episode_rewards']
    episode_lengths = data['episode_lengths']
    title = data_path.split('/')[-2]
    
    print(f"Loaded {len(episode_rewards)} episodes")
    print(f"Total timesteps: {np.sum(episode_lengths):,}")
    
    if use_episodes:
        # Plot using actual episode data (more accurate)
        episode_timesteps = compute_episode_timesteps(episode_lengths)
        rolling_avg = compute_rolling_average(episode_rewards, window=100)
        
        # Smooth the rolling average
        smoothed_rewards = smooth_curve(rolling_avg, window_size=smooth_window)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot individual episodes (very faint)
        if show_raw:
            plt.scatter(episode_timesteps, episode_rewards, s=1, alpha=0.1, 
                       color='gray', label='Individual Episodes', rasterized=True)
        
        # Plot rolling average (semi-transparent)
        # plt.plot(episode_timesteps, rolling_avg, linewidth=1, alpha=0.4, 
        #         color='orange', label='Rolling Avg (100 episodes)')
        
        # Plot smoothed rolling average
        plt.plot(episode_timesteps, smoothed_rewards, linewidth=1, color='blue', 
                label=f'Rewards')
        
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Episode Reward', fontsize=12)
        plt.title(f'Training Progress: Average Reward vs Timesteps - {title}', fontsize=14)
        
    else:
        # Plot using update-based data (original method)
        timesteps = data['timesteps']
        avg_rewards = data['avg_rewards']
        
        # Smooth the rewards curve
        smoothed_rewards = smooth_curve(avg_rewards, window_size=smooth_window)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot raw data (optional, with transparency)
        if show_raw:
            plt.plot(timesteps, avg_rewards, linewidth=1, alpha=0.3, 
                    color='blue', label='Update-based Average')
        
        # Plot smoothed data
        plt.plot(timesteps, smoothed_rewards, linewidth=2.5, color='blue', 
                label=f'Smoothed Reward (window={smooth_window})')
        
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title(f'Training Progress: Average Reward vs Timesteps - {title}', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    
    # Format x-axis to show thousands
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))

    # plt.ylim(-500, 100)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(f"{save_path}_{title}.png", dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()

def formatter(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.1f}K'
    else:
        return f'{x:.0f}'

def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("data_path", type=str, help="Path to training_data.npz file")
    parser.add_argument("--save", type=str, default=None, help="Path to save plot (optional)")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size (default: 10)")
    parser.add_argument("--show-raw", action="store_true", help="Show raw data along with smoothed curve")
    parser.add_argument("--episodes", action="store_true", help="Use episode data for accurate timestep plotting")
    
    args = parser.parse_args()
    
    # Auto-generate save path if not provided
    save_path = args.save
    if save_path is None:
        data_dir = Path(args.data_path).parent
        suffix = "_episodes" if args.episodes else "_updates"
        save_path = str(data_dir / f"training_plot{suffix}")
    
    plot_training_data(args.data_path, save_path, smooth_window=args.smooth, 
                      show_raw=args.show_raw, use_episodes=args.episodes)

if __name__ == "__main__":
    main()
