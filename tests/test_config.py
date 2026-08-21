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
kl_early_stop = True
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
    assert cfg.kl_early_stop is True  # only in the task file
    assert cfg.gamma == 0.98  # from defaults
    assert cfg.hidden_dims == [256, 128, 64]  # from defaults
