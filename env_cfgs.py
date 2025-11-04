class EnvConfig:
    # Configuration parameters for PPO training in different IsaacLab environments
    def __init__(self, args_cli):
        self.args_cli = args_cli

        # Default parameters
        self.num_steps_per_env = 24
        self.num_mini_batches = 4
        self.num_learning_epochs = 8
        self.max_iterations = 1000
        self.lr = 3e-4
        self.hidden_dims = [64, 64]
        self.value_coef = 1.0
        self.entropy_coef = 1e-3
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.schedule_type = "adaptive"
        self.desired_kl = 0.01
        self.max_grad_norm = 0.5
        self.gamma = 0.99

        # Task-specific overrides
        if "Cartpole" in args_cli.task:
            self.num_steps_per_env = 16
            self.num_learning_epochs = 5
            self.max_iterations = 400
            self.hidden_dims = [32, 32]
            self.entropy_coef = 5e-3
        elif "Lift" in args_cli.task:
            self.lr=1e-4
            self.hidden_dims = [256, 128, 64]
            self.num_learning_epochs = 5
            self.max_iterations = 1500
            self.entropy_coef = 6e-3
        elif "Repose" in args_cli.task:
            self.lr = 1e-3
            self.num_learning_epochs = 5
            self.max_iterations = 5000
            self.gamma = 0.998
            self.hidden_dims = [512, 256, 128]
            self.entropy_coef = 2e-3
        elif "Drawer" in args_cli.task:
            self.lr = 5e-4
            self.hidden_dims = [256, 128, 64]
            self.num_steps_per_env = 96
            self.num_mini_batches = 8
            self.num_learning_epochs = 5
            self.max_iterations = 400
            self.entropy_coef = 1e-3