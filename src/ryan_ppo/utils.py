from __future__ import annotations

import random
import sys
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import numpy as np
import rich.traceback
import torch
from rich import box
from rich.table import Table

# libraries to suppress in rich traceback.
SUPPRESSED_LIBRARIES = ("gymnasium",)


def install_rich_traceback(show_locals: bool = False):
    """
    initialize the rich traceback, including suppressing specified libaries.
    """
    # every isaaclab package, when running against isaaclab at all
    suppress = []
    isaaclab = sys.modules.get("isaaclab")
    if isaaclab is not None:
        suppress.append(str(Path(isaaclab.__file__).parents[2]))

    suppress += [import_module(name) for name in SUPPRESSED_LIBRARIES]

    rich.traceback.install(suppress=suppress, show_locals=show_locals)
    return sys.excepthook


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    return device


class PhaseTimer:
    """Time a phase on its CUDA stream, or with a CPU clock on CPU.

    Read seconds() after the update's existing synchronization, so recording
    phase boundaries does not introduce synchronization between phases.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self._seconds = 0.0

    def start(self) -> None:
        if self.cuda:
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self._start = time.perf_counter()

    def stop(self) -> None:
        if self.cuda:
            self.end_event.record(torch.cuda.current_stream(self.device))
        else:
            self._seconds = time.perf_counter() - self._start

    def seconds(self) -> float:
        if self.cuda:
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        return self._seconds


def set_seed(seed: int) -> None:
    # seed python, numpy and torch, and pin cudnn to deterministic kernels.
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_dims(env) -> tuple[int, int]:
    # (state_dim, action_dim), for both Dict and Box observation spaces.
    obs_space = env.observation_space
    spaces = getattr(obs_space, "spaces", None)
    if spaces is not None:
        obs_space = spaces["policy"]
    return obs_space.shape[1], env.action_space.shape[1]


class Profiler:
    """carb.profiler zones, or no-ops when profiling is off.

    keeps the carb import and the `if args_cli.profile` guards out of train().
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._carb = None
        if enabled:
            import carb.profiler

            self._carb = carb.profiler

    def begin(self, zone_id: int, name: str) -> None:
        if self._carb:
            self._carb.begin(zone_id, name)

    def end(self, zone_id: int) -> None:
        if self._carb:
            self._carb.end(zone_id)

    @contextmanager
    def zone(self, zone_id: int, name: str):
        self.begin(zone_id, name)
        try:
            yield
        finally:
            self.end(zone_id)


def policy_obs(state):
    # returns observation regardless of if it is in a dict or a box.
    if isinstance(state, dict):
        return state.get("policy", next(iter(state.values())))
    return state


def get_cfg_path(task):
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    ini_file_path = project_root / "cfg" / f"{task}.ini"
    if not ini_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {ini_file_path}")

    return ini_file_path


def generate_table(
    perf_dict: dict,
    train_dict: dict,
    rewards_dict: dict,
    title: str,
    run_url: str | None = None,
) -> Table:
    """
    Make table for display in terminal of current training run.
    """

    TITLE_COLOR = "[dodger_blue1]"
    if run_url:
        title = f"[link={run_url}]{title} (WandB)[/link]"

    # outer table, allows title within table.
    main_table = Table(
        box=box.ROUNDED,
        show_header=False,
        border_style="magenta",
    )
    main_table.add_column(justify="center")
    main_table.add_row(TITLE_COLOR + title)
    main_table.add_section()

    # performance metrics (throughput / progress) and training metrics
    perf_table = build_metric_table("Performance", perf_dict)
    train_table = build_metric_table("Training", train_dict)

    # reward terms and average values over recent episodes.
    reward_table = Table(box=None, show_edge=False, border_style="cyan", expand=True)
    reward_table.add_column(TITLE_COLOR + "Reward Terms", style="white", width=40)
    reward_table.add_column(
        TITLE_COLOR + "Value", justify="right", style="white", width=12
    )

    # adds and formats rewards within the rewards_dict passed in.
    for key, value in rewards_dict.items():
        reward_table.add_row(key, format_values(key, value))

    # stack the two metric tables vertically on the left.
    left_stack = Table.grid(padding=(1, 0))
    left_stack.add_row(perf_table)
    left_stack.add_row(train_table)

    # reward terms on the right.
    right_stack = Table.grid(padding=(1, 0))
    right_stack.add_row(reward_table)

    inner_grid = Table.grid(padding=4)
    inner_grid.add_row(left_stack, right_stack)

    main_table.add_row(inner_grid)

    return main_table


def build_metric_table(header: str, stats_dict: dict) -> Table:
    """
    Build an inner two-column table (metric name, value) under a header.
    """

    TITLE_COLOR = "[dodger_blue1]"
    table = Table(box=None, show_edge=False, border_style="cyan", expand=True)
    table.add_column(TITLE_COLOR + header, style="white", width=14)
    table.add_column(TITLE_COLOR + "Value", justify="right", style="white", width=12)

    for key, value in stats_dict.items():
        table.add_row(key, format_values(key, value))

    return table


def format_values(name, value):

    if name == "Runtime" or name == "Remaining Time":
        if value > 60:
            formatted_value = f"{value // 60:.0f}m {value % 60:.0f}s"
        else:
            formatted_value = f"{value:.0f}s"
        return formatted_value
    elif name in ("Rollout Time", "Preparation Time", "Update Time"):
        if value >= 1:
            return f"{value:.2f}s"
        return f"{value * 1000:.0f}ms"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        abs_val = abs(value)
        if abs_val >= 1_000_000:
            formatted_value = f"{value / 1_000_000:.2f}M"
        elif abs_val >= 1_000:
            formatted_value = f"{value / 1_000:.2f}K"
        else:
            formatted_value = f"{value:.6f}".rstrip("0").rstrip(".")
            if not formatted_value:
                formatted_value = "0"
        return formatted_value
    else:
        return str(value)
