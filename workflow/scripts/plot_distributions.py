from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from natural_hazard_solidarity.plotting import set_thesis_style
set_thesis_style()

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake


def paired_likert(df: pd.DataFrame, pre_col: str, post_col: str) -> tuple[np.ndarray, np.ndarray]:
    values = df[[pre_col, post_col]].copy()
    values[pre_col] = pd.to_numeric(values[pre_col], errors="coerce")
    values[post_col] = pd.to_numeric(values[post_col], errors="coerce")
    values = values.dropna()
    values = values[values[pre_col].between(1, 6) & values[post_col].between(1, 6)]
    return values[pre_col].astype(int).to_numpy(), values[post_col].astype(int).to_numpy()


def plot_figure(df: pd.DataFrame, specification: dict) -> None:
    items = specification["items"]
    labels = specification.get("labels", {})
    n_items = len(items)
    ncols = 1 if n_items == 1 else 3 if n_items == 3 else 2
    nrows = ceil(n_items / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.2 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    bins = np.arange(0.5, 7.0, 1.0)
    levels = np.arange(1, 7)
    legend_handles = None
    legend_labels = None

    for index, (axis, item) in enumerate(zip(axes, items)):
        pre_col = f"S0_{item}"
        post_col = f"S1_{item}"
        if pre_col not in df.columns or post_col not in df.columns:
            axis.set_title(f"{labels.get(item, item)} (missing)")
            axis.axis("off")
            continue

        pre, post = paired_likert(df, pre_col, post_col)
        if len(pre) == 0:
            axis.set_title(f"{labels.get(item, item)} (no paired data)")
            axis.axis("off")
            continue

        axis.hist(pre, bins=bins, alpha=0.45, label="Pre")
        axis.hist(post, bins=bins, alpha=0.45, label="Post")
        axis.set_title(labels.get(item, item))
        axis.set_xticks(levels)
        axis.set_xlim(0.5, 6.5)
        if index % ncols == 0:
            axis.set_ylabel("Respondents")
        if index // ncols == nrows - 1:
            axis.set_xlabel("Likert response")
        else:
            axis.tick_params(labelbottom=False)

        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    for axis in axes[n_items:]:
        axis.axis("off")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            frameon=False,
        )
    fig.suptitle(specification.get("title", ""), y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    output = Path(specification["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


df = pd.read_parquet(snakemake.input.analysis)
for figure in snakemake.params.figures:
    plot_figure(df, figure)
