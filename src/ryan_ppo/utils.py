from rich import box
from rich.table import Table


def generate_table(
    stats_dict: dict, rewards_dict: dict, title: str, run_url: str = None
) -> Table:
def generate_table(
    stats_dict: dict, rewards_dict: dict, title: str, run_url: str = None
) -> Table:
    """
    Make table for display in terminal of current training run.
    """

    TITLE_COLOR = "[dodger_blue1]"

    if run_url:
        title = f"[link={run_url}]{title} (WandB)[/link]"

    TITLE_COLOR = "[dodger_blue1]"

    if run_url:
        title = f"[link={run_url}]{title} (WandB)[/link]"

    # outer table, allows title within table.
    main_table = Table(
        box=box.ROUNDED,
        show_header=False,
        border_style="magenta",
        border_style="magenta",
    )
    main_table.add_column(justify="center")
    main_table.add_row(TITLE_COLOR + title)
    main_table.add_row(TITLE_COLOR + title)
    main_table.add_section()

    # first inner table, contains metrics and values for training data.
    metric_table = Table(box=None, show_edge=False, border_style="cyan", expand=True)
    metric_table = Table(box=None, show_edge=False, border_style="cyan", expand=True)

    # second inner table, contains reward terms and average values over recent episodes.
    reward_table = Table(box=None, show_edge=False, border_style="cyan", expand=True)
    reward_table = Table(box=None, show_edge=False, border_style="cyan", expand=True)

    metric_table.add_column(TITLE_COLOR + "Run Info", style="white", width=12)
    metric_table.add_column(
        TITLE_COLOR + "Value", justify="right", style="white", width=12
    )

    reward_table.add_column(TITLE_COLOR + "Reward Terms", style="white", width=40)
    reward_table.add_column(
        TITLE_COLOR + "Value", justify="right", style="white", width=12
    )

    # adds and formats metrics within the stats_dict passed in.
    for key, value in stats_dict.items():
        metric_table.add_row(key, format_values(key, value))

    # adds and formats rewards within the rewards_dict passed in.
    for key, value in rewards_dict.items():
        reward_table.add_row(key, format_values(key, value))

    inner_grid = Table.grid(padding=4)
    inner_grid.add_row(metric_table, reward_table)

    main_table.add_row(inner_grid)

    return main_table


def format_values(name, value):

    if name == "Runtime" or name == "Remaining Time":
        if value > 60:
            formatted_value = f"{value // 60:.0f}m {value % 60:.0f}s"
        else:
            formatted_value = f"{value:.0f}s"
        return formatted_value
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
