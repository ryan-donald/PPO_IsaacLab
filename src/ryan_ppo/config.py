import configparser
from dataclasses import dataclass


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
    saturation_coef: float = 1e-3

    @classmethod
    def from_ini(cls, path):
        config = configparser.ConfigParser()
        config.read(path)
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
            saturation_coef=train.getfloat("saturation_coef", fallback=1e-3),
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
