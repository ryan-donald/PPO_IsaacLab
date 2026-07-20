[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/whatsnew/3.12.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)
![Tests](https://github.com/ryan-donald/ppo/actions/workflows/tests.yaml/badge.svg)

# PPO for IsaacLab
This is a repository containing my implementation of the Proximal Policy Optimization (PPO) Reinforcement Learning algorithm, specifically for use in Nvidia's IsaacLab. I initially developed and tested this algorithm within gymnasium, and then moved to IsaacLab. The base algorithm is not specific to the environment, and will work with any environment as long as the batch data is in the expected format.

<img width="720" height="405" alt="so101_reach" src="https://github.com/user-attachments/assets/a04e37d7-f6f0-4f09-af24-27157c920124" />

# Benchmarks
I benchmarked this implementation against the four RL libraries bundled with Isaac Lab — [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [rl_games](https://github.com/Denys88/rl_games), [skrl](https://github.com/Toni-SM/skrl), and [sb3](https://github.com/DLR-RM/stable-baselines3) — on the `Ryan-Reach-SO-ARM101-Normalized-v0` task. Every run used identical settings on the same GPU (RTX 3070): 12,288 parallel environments, 4,000 iterations, 24 steps/env, seed 42, headless. The library agent configs are hyperparameter-matched to `ryan_ppo`.

| Framework | Throughput (steps/s) | In-loop steps/s¹ | Wall-clock (min) | Peak reward | Time to competent policy² |
|---|---:|---:|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,320,001** | **1,208,656** | **15.7** | **1.094** | 2.4 min |
| rl_games | 1,077,524 | 1,092,267 | 18.9 | 0.929 | 2.4 min |
| skrl | 1,061,668 | 1,084,235 | 19.2 | 0.123 | — |
| rsl_rl | 834,752 | 845,020 | 24.2 | 0.570 | 2.8 min |
| sb3 | 619,342 | — | 32.4 | 0.572 | 4.7 min |

<sub>¹ Rollout + full 5-epoch update per iteration, excluding logging/checkpoint overhead. `ryan_ppo`'s update time counts only iterations that ran all 5 epochs, so KL early-stopping isn't credited. sb3 logs no rollout/update split.<br>² Wall-clock to first reach a mean episode reward of 0.5, indicating a competent reach policy. skrl never reached it.</sub>

This implementation delivers the **highest training throughput** — ~22% faster end-to-end than the next-quickest library and 2.1× faster than SB3 — the **fastest wall-clock** time to complete the run, and the **highest peak reward** of any framework tested. It reaches a competent policy as quickly as the fastest library while converging to a substantially higher peak.

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_reward_vs_time.png" width="100%" alt="Reward vs wall-clock time" />
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_reward_vs_steps.png" width="100%" alt="Reward vs environment steps" />
</div>

# Quickstart
To use this package, follow the steps below:

* Install and setup Nvidia's IsaacLab, found [here](https://github.com/isaac-sim/IsaacLab).
* Install my custom IsaacLab tasks from [tasks-isaaclab](https://github.com/ryan-donald/tasks-isaaclab), which provides the `ryan_tasks` package the training scripts import: clone it and run "pip install -e source/ryan_tasks" in its base directory.
* Clone this repository.
* Run the command "pip install -e ." within this repository.
* You are all set and can now train agents within IsaacLab using this package. An example training run command is below:
* "python -m ryan_ppo.isaaclab.train --task Ryan-Reach-SO-ARM101-Normalized-v0 --num_envs 2048 --headless".

# Features
Fully functional PPO agent, with a configuration file where you can set hyperparameters depending on the task you are running. Additionally, training runs are tracked and stored utilizing Weights and Biases, allowing for easy performance tracking and comparison between runs. 

## Multiple Environments
The base algorithm, defined in the files within the 'src/' directory, are portable to any gym-style environment. Within the 'src/' directory is an 'isaaclab/' directory containing a *train.py* and *play.py* file, which implement the algorithm specifically for IsaacLab. To use the algorithm in another set of environments, simply create your own *train.py* and *play.py* files for those environments in this format. 

## Weights and Biases Parameter Sweeping
This implementation supports parameter sweeping via Weights and Biases. To do this, create a YAML description file in the format of those in "cfg/sweeps/". Within this file, define either a set of discrete values or a distribution for each parameter that you want to be swept. Ensure that *train.py* contains checks for all of the parameters that are being swept to ensure they are actually being used in the runs. After this, run "wandb sweep <sweep config file>" followed by "wandb agent <username>/<project name>/<sweep id>". The results will be logged via Weights and Biases. Shown below is an example plot showing 50 different runs with a reach task, sweeping over a handful of parameters.

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/so101_reach_sweep.png" width="100%" alt="Parameter Sweep">
</div>

## Sim2Real
Using this package, I have been able to perform Sim2Real transfer of a Reach agent for the open source SO-ARM101 robot. Specifics about that process can be found [here](https://ryan-donald.github.io/portfolio/1-PPO_Sim2Real/), and my script can be found [here](https://github.com/ryan-donald/so101_ppo).

[![PPO SO-ARM101 sim2real](https://img.youtube.com/vi/MzxyW7mrM0s/maxresdefault.jpg)](https://www.youtube.com/watch?v=MzxyW7mrM0s)

## Experiment Tracking in Terminal
With the help of the python package [rich](https://github.com/textualize/rich), I have a display in the terminal which provides information about the currently running experiment, including reward terms, learning parameters, performance, remaining time, and a clickable link to the current WandB run. An example of this can be seen below:

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/terminal_display.png" width="100%" alt="Parameter Sweep">
</div>

## Training Run Profiling with Tracy Profiler
Based on the recommendation in the official Isaac Lab documentation [here](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/utilities/debugging/profiling_performance.html), I added support for code profiling using the Tracy profiler. This allows for live profiling of the performance of the training script. This will provide information for the time spent in each block of execution provide information that can be used to gauge and improve the efficiency of the training script, and various environments. To use this, simply add these flags to the script: "--profile --enable omni.kit.profiler.tracy".

# Script Structure
The main structure of this repository is as follows:  
* network.py - Contains the *Actor* and *Critic* network classes, used to represent the policy and value functions in the PPO algorithm.  
  
* ppo.py - Contains the *PPOAgent* class, which stores and configures the various hyper-parameters of the algorithm, as well as the following functions:  
  * *select_action* - Selects an action based upon an observation and the current policy.  
  * *compute_gae* - Computes the normalized Generalized Advantage Estimates at each step of the roll-out, as well as the returns.  
  * *update* - Performs the update portion of the PPO algorithm. Given a rollout, this function performs a number of updates, each with a number of mini-batches from the main rollout data.  

* train.py - Contains the training loop, any initialization code, and calls functions to implement the entire PPO algorithm.  
  
* config.py - Parses per-task hyper-parameter files from the 'cfg/' directory. Shared values live in 'cfg/defaults.ini'; each task's .ini file only sets the values that differ from those defaults.
