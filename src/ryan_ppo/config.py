import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """parses config file for tasks."""

    learning_rate: float
    gamma: float
    gae_lambda: float
    value_coef: float
    clip_epsilon: float
    max_grad_norm: float
    desired_kl: float
    entropy_coef: float
    schedule_type: str
    num_learning_epochs: int
    num_steps_per_env: int
    num_mini_batches: int
    max_iterations: int
    use_normalization: bool
    hidden_dims: list[int]
    max_lr: float = 1e-3
    min_lr: float = 1e-5
    std_min: float = 0.005
    std_max: float = 1.0
    kl_early_stop: bool = False
    stagger_initial_episodes: bool = True

    @classmethod
    def from_ini(cls, path):
        # allows reading a default config then a override, or a full config.
        path = Path(path)
        config = configparser.ConfigParser()
        config.read([path.parent / "defaults.ini", path])
        train = config["train"]
        return cls(
            learning_rate=train.getfloat("learning_rate"),
            gamma=train.getfloat("gamma"),
            gae_lambda=train.getfloat("gae_lambda"),
            value_coef=train.getfloat("value_coef"),
            clip_epsilon=train.getfloat("clip_epsilon"),
            max_grad_norm=train.getfloat("max_grad_norm"),
            desired_kl=train.getfloat("desired_kl"),
            entropy_coef=train.getfloat("entropy_coef"),
            schedule_type=train["schedule_type"],
            num_learning_epochs=train.getint("num_learning_epochs"),
            num_steps_per_env=train.getint("num_steps_per_env"),
            num_mini_batches=train.getint("num_mini_batches"),
            max_iterations=train.getint("max_iterations"),
            use_normalization=train.getboolean("use_normalization"),
            hidden_dims=[int(x) for x in config["policy"]["hidden_dims"].split(",")],
            max_lr=train.getfloat("max_lr", fallback=1e-3),
            min_lr=train.getfloat("min_lr", fallback=1e-5),
            std_min=train.getfloat("std_min", fallback=0.005),
            std_max=train.getfloat("std_max", fallback=1.0),
            kl_early_stop=train.getboolean("kl_early_stop", fallback=False),
            stagger_initial_episodes=train.getboolean(
                "stagger_initial_episodes", fallback=True
            ),
        )

    def apply_sweep(self, sweep):
        # applys values from wandb sweep to current config.
        overrides = {
            "lr": "learning_rate",
            "gamma": "gamma",
            "num_learning_epochs": "num_learning_epochs",
            "desired_kl": "desired_kl",
            "clip_epsilon": "clip_epsilon",
            "entropy_coef": "entropy_coef",
        }
        for key, attr in overrides.items():
            if key in sweep:
                setattr(self, attr, sweep[key])
