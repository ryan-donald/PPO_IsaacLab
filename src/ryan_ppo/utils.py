from rich import box
from rich.table import Table


def policy_obs(state):
    # returns observation regardless of if it is in a dict or a box.
    if isinstance(state, dict):
        return state.get("policy", next(iter(state.values())))
    return state


def generate_table(
    perf_dict: dict,
    train_dict: dict,
    rewards_dict: dict,
    title: str,
    run_url: str = None,
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
    elif name == "Rollout Time" or name == "Update Time":
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
