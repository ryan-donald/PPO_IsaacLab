import configparser
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

# configparser reader per field type.
TYPE_READERS = {
    "float": "getfloat",
    "int": "getint",
    "bool": "getboolean",
    "str": "get",
}


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
    std_init: float = 0.5
    stagger_initial_episodes: bool = True

    @classmethod
    def from_ini(cls, path):
        # allows reading a default config then a override, or a full config.
        path = Path(path)
        config = configparser.ConfigParser()
        config.read([path.parent / "defaults.ini", path])
        train = config["train"]

        kwargs = {}
        for field in fields(cls):
            if field.name == "hidden_dims":
                continue
            type_name = getattr(field.type, "__name__", field.type)
            read = getattr(train, TYPE_READERS[type_name])
            if field.default is not MISSING:
                kwargs[field.name] = read(field.name, fallback=field.default)
            elif field.name in train:
                kwargs[field.name] = read(field.name)
            else:
                raise KeyError(
                    f"{path}: required [train] key {field.name!r} is missing from "
                    f"both the task file and defaults.ini"
                )

        hidden_dims = [int(x) for x in config["policy"]["hidden_dims"].split(",")]
        return cls(hidden_dims=hidden_dims, **kwargs)

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
