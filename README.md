[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/whatsnew/3.12.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)
![Tests](https://github.com/ryan-donald/ppo/actions/workflows/tests.yaml/badge.svg)

# PPO for IsaacLab
This is a repository containing my implementation of the Proximal Policy Optimization (PPO) Reinforcement Learning algorithm, specifically for use in Nvidia's IsaacLab. I initially developed and tested this algorithm within gymnasium, and then moved to IsaacLab. The base algorithm is not specific to the environment, and will work with any environment as long as the batch data is in the expected format.

<img width="720" height="405" alt="so101_reach" src="https://github.com/user-attachments/assets/a04e37d7-f6f0-4f09-af24-27157c920124" />

# Benchmarks
I benchmarked this implementation against the four RL libraries bundled with Isaac Lab — [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [rl_games](https://github.com/Denys88/rl_games), [skrl](https://github.com/Toni-SM/skrl), and [sb3](https://github.com/DLR-RM/stable-baselines3) — across three tasks of increasing difficulty: `Ryan-Cartpole-v0`, `Ryan-Ant-v0`, and `Ryan-Reach-SO-ARM101-Normalized-v0`. Every run used identical settings on the same GPU (RTX 3070) under Isaac Lab 3.0: 12,288 parallel environments, headless, and each library's agent config hyperparameter-matched to `ryan_ppo`. Each framework was run over **three seeds (42, 43, 44)**; the tables report the seed-averaged best mean episode reward reached during training, since the best checkpoint is the one deployed.

**Cartpole** — 1,000 iterations, 16 steps/env

| Framework | Throughput (steps/s) | Wall-clock (min) | Best reward |
|---|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,243,830** | **3.0** | 4.961 |
| rl_games | 1,093,053 | 3.3 | 4.962 |
| skrl | 1,073,521 | 3.3 | 4.956 |
| rsl_rl | 1,033,656 | 3.4 | 4.958 |
| sb3 | 666,278 | 5.2 | 4.297 |

**Ant** — 2,000 iterations, 32 steps/env

| Framework | Throughput (steps/s) | Wall-clock (min) | Best reward |
|---|---:|---:|---:|
| **ryan_ppo (this repo)** | **624,188** | **21.6** | **136.6** |
| rsl_rl | 556,776 | 24.1 | 136.1 |
| skrl | 555,062 | 24.2 | 114.6 |
| rl_games | 554,338 | 24.2 | 129.1 |
| sb3 | 387,591 | 34.4 | 134.3 |

**Reach** — 7,500 iterations, 24 steps/env

| Framework | Throughput (steps/s) | Wall-clock (min) | Best reward |
|---|---:|---:|---:|
| **ryan_ppo (this repo)** | **1,283,631** | **29.5** | **0.927** |
| skrl | 1,084,497 | 34.7 | 0.861 |
| rl_games | 1,072,870 | 35.1 | 0.900 |
| rsl_rl | 844,198 | 44.3 | 0.640 |
| sb3 | 627,026 | 59.5 | 0.727 |

Across all three tasks, `ryan_ppo` posts the **highest throughput and fastest wall-clock** of any framework tested — 12–18% ahead of the next-quickest library, and up to 2× faster end-to-end than SB3. Its reward is **at or above the best of any library** on every task: it ties the field on Cartpole, edges rsl_rl for the top spot on Ant (while finishing several minutes sooner), and reaches the highest reward on Reach. On Reach it also *holds* that peak — where the other frameworks peak mid-run and then regress toward ~0.6, `ryan_ppo` stays near its best through the end of training (final ≈ 0.91 vs best ≈ 0.93).

<div align="center">
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_cartpole_reward_vs_time.png" width="100%" alt="Cartpole reward vs wall-clock time" />
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_ant_reward_vs_time.png" width="100%" alt="Ant reward vs wall-clock time" />
  <img src="https://raw.githubusercontent.com/ryan-donald/ppo/main/images/benchmark_reach_reward_vs_time.png" width="100%" alt="Reach reward vs wall-clock time" />
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
