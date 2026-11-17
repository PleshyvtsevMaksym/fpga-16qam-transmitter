"""Report 02 page: A4 landscape page with parameters, plot, values and description."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAGE_SIZE = (11.69, 8.27)       # A4 landscape, inches


def new_page(
    number: str,
    title: str,
    stamp: str,
    parameter_columns: list[str],
    values: dict,
    description: str
):
    """
    Create one report page and return it with an empty plot area.

    Layout: title and run stamp, parameter block of the run, plot (left),
    measured values (right), description (bottom).

    Parameters
    ----------
    number : str
        Page number in chain order, e.g. "03-2".
    title : str
        What the plot shows.
    stamp : str
        Version, date and commit (run_stamp).
    parameter_columns : list of str
        Text columns from parameter_columns().
    values : dict
        Measured values, {"name": "value with unit"}.
    description : str
        What is shown, how to read it, what it depends on.

    Returns
    -------
    fig, ax
        The page and its plot axes.
    """

    fig = plt.figure(figsize=PAGE_SIZE, layout="none")

    # 1. Header
    fig.text(0.04, 0.962, f"{number}   {title}", fontsize=13, weight="bold", va="center")
    fig.text(0.96, 0.962, stamp, fontsize=7.5, color="0.35", ha="right", va="center")
    fig.add_artist(Line2D([0.04, 0.96], [0.935, 0.935], color="0.2", linewidth=0.8))

    # 2. Parameters of the run
    for index, column in enumerate(parameter_columns):
        fig.text(0.04 + index * 0.235, 0.922, column, fontsize=6.4, family="monospace",
                 va="top", linespacing=1.25)
    fig.add_artist(Line2D([0.04, 0.96], [0.79, 0.79], color="0.75", linewidth=0.6))

    # 3. Plot area
    ax = fig.add_axes([0.075, 0.2, 0.6, 0.55])

    # 4. Measured values
    fig.text(0.715, 0.75, "MEASURED VALUES", fontsize=8.5, weight="bold", va="top")
    width = max(len(name) for name in values) if values else 0
    lines = [f"{name:<{width}}  {value}" for name, value in values.items()]
    fig.text(0.715, 0.72, "\n".join(lines), fontsize=7.6, family="monospace", va="top",
             linespacing=1.45)

    # 5. Description
    fig.add_artist(Line2D([0.04, 0.96], [0.115, 0.115], color="0.75", linewidth=0.6))
    fig.text(0.04, 0.1, textwrap.fill(description, 175), fontsize=7.8, va="top",
             linespacing=1.35)

    return fig, ax
