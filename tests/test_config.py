import dataclasses

import pytest

from ryan_ppo.config import TrainConfig

TASK_INI = """
[train]
num_steps_per_env = 24
num_mini_batches = 4
num_learning_epochs = 5
max_iterations = 15000
learning_rate = 0.0005
value_coef = 1.0
entropy_coef = 0.01
gae_lambda = 0.95
clip_epsilon = 0.2
schedule_type = adaptive
desired_kl = 0.016
max_grad_norm = 1.0
gamma = 0.98
use_normalization = True

[policy]
hidden_dims = 256,128,64
"""

DEFAULTS_INI = """
[train]
num_steps_per_env = 24
num_mini_batches = 4
num_learning_epochs = 5
max_iterations = 15000
learning_rate = 0.0005
value_coef = 1.0
entropy_coef = 0.01
gae_lambda = 0.95
clip_epsilon = 0.2
schedule_type = adaptive
desired_kl = 0.016
max_grad_norm = 1.0
gamma = 0.98
use_normalization = True

[policy]
hidden_dims = 256,128,64
"""

DELTA_INI = """
[train]
learning_rate = 0.001
std_min = 0.05
"""


def test_from_ini(tmp_path):
    path = tmp_path / "task.ini"
    path.write_text(TASK_INI)

    cfg = TrainConfig.from_ini(path)
    assert cfg.learning_rate == 0.0005
    assert cfg.gamma == 0.98
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 15000
    assert cfg.schedule_type == "adaptive"
    assert cfg.use_normalization is True
    assert cfg.hidden_dims == [256, 128, 64]


def test_from_ini_layers_defaults(tmp_path):
    # a task file only needs its deltas; defaults.ini beside it fills the rest.
    (tmp_path / "defaults.ini").write_text(DEFAULTS_INI)
    path = tmp_path / "task.ini"
    path.write_text(DELTA_INI)

    cfg = TrainConfig.from_ini(path)
    assert cfg.learning_rate == 0.001  # overridden by the task file
    assert cfg.std_min == 0.05  # only in the task file
    assert cfg.gamma == 0.98  # from defaults
    assert cfg.hidden_dims == [256, 128, 64]  # from defaults


def test_from_ini_missing_required_key_raises(tmp_path):
    # a required [train] key absent from both files is a config error, not a
    # silent None that blows up later inside PPOAgent.
    path = tmp_path / "task.ini"
    path.write_text("[train]\ngamma = 0.99\n\n[policy]\nhidden_dims = 8,8\n")

    with pytest.raises(KeyError, match="learning_rate"):
        TrainConfig.from_ini(path)


def test_from_ini_reads_every_field(tmp_path):
    # from_ini drives off dataclasses.fields(), so a new field must be picked up
    # without touching from_ini. this guards that wiring.
    path = tmp_path / "task.ini"
    extra = "max_lr = 0.02\nstagger_initial_episodes = False\n"
    path.write_text(TASK_INI.replace("\n[policy]", f"{extra}\n[policy]"))

    cfg = TrainConfig.from_ini(path)
    for field in dataclasses.fields(TrainConfig):
        assert getattr(cfg, field.name) is not None, field.name
    assert cfg.max_lr == 0.02
    assert cfg.stagger_initial_episodes is False
    assert cfg.min_lr == 1e-5  # untouched optional field keeps its default


def test_apply_sweep(tmp_path):
    path = tmp_path / "task.ini"
    path.write_text(TASK_INI)
    cfg = TrainConfig.from_ini(path)
    cfg.apply_sweep({"lr": 0.001, "num_learning_epochs": 3})
    assert cfg.learning_rate == 0.001
    assert cfg.num_learning_epochs == 3
    assert cfg.gamma == 0.98
