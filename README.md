This is a repository containing my implementation of PPO, specifically for use in Nvidia's IsaacLab. I initially developed this algorithm for use with Gymnasium, and this implementation could be utilized with Gymnasium or other environments, mainly with changes needed in the *train.py* file.  
  
The main structure of this repository is as follows:  
* network.py - Contains the *Actor* and *Critic* network classes, used to represent the policy and value functions in the PPO algorithm.  
  
* ppo.py - Contains the *PPOAgent* class, which stores and configures the various hyper-parameters of the algorithm, as well as the following functions:  
  * *select_action* - Selects an action based upon an observation and the current policy.  
  * *compute_gae* - Computes the normalized Generalized Advantage Estimates at each step of the roll-out, as well as the returns.  
  * *update* - Performs the update portion of the PPO algorithm. Given a rollout, this function performs a number of updates, each with a number of mini-batches from the main rollout data.  

* train.py - Contains the training loop, any initialization code, and calls functions to implement the entire PPO algorithm.  
  
* env_cfgs.py - Contains various hyper-parameters for PPO depending on the environment within IsaacLab that is being utilized.
