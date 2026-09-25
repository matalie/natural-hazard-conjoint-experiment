"""Create descriptive figures that summarize observed survey responses,
distributions, and pre/post patterns before model-based inference."""
from __future__ import annotations

from math import ceil
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import(PURPLE, LIGHT_PURPLE, TEAL, BROWN)

LIKERT_VALUES = (1, 2, 3, 4, 5, 6)
LIKERT_BINS = np.arange(0.5, 7.0, 1.0)

DESERVINGNESS_COLUMNS = [
    "S1_deservingness_1",
    "S1_deservingness_3",
    "S1_deservingness_4",
    "S1_deservingness_7",
    "S1_deservingness_8",
    "S1_deservingness_9",
    "S1_deservingness_10",
    "S1_deservingness_11",
    "S1_deservingness_12",
]

DESERVINGNESS_TITLES = {
    "S1_deservingness_1": "Municipalities in which people have lived in for many years",
    "S1_deservingness_3": "Municipalities that knowingly built in hazard risk zones",
    "S1_deservingness_4": "Municipalities that build in areas, which became later designated as hazard risk zones",
    "S1_deservingness_7": "Municipalities with wealthy residents who contribute more to federal taxes and natural hazard prevention",
    "S1_deservingness_8": "Municipalities with lower income residents who contribute less to federal taxes and natural hazard prevention",
    "S1_deservingness_9": "Municipalities that do not implement climate protection measures (e.g. investing in renewable energy)",
    "S1_deservingness_10": "Municipalities that implement climate protection measures (e.g. investing in renewable energy)",
    "S1_deservingness_11": "Municipalities with high risks of natural hazards",
    "S1_deservingness_12": "Municipalities with low risks of natural hazards",
}


def paired_likert(df: pd.DataFrame, pre_col: str, post_col: str):
    values = df[[pre_col, post_col]].apply(pd.to_numeric, errors="coerce").dropna()
    values = values[values[pre_col].between(1, 6) & values[post_col].between(1, 6)]
    return values[pre_col].astype(int).to_numpy(), values[post_col].astype(int).to_numpy()


def plot_pre_post_likert(df: pd.DataFrame, spec: dict):
    """Plot S0/S1 (pre/post) Likert histograms for one construct group."""
    items = list(spec["items"])
    labels = dict(spec.get("labels", {}))
    k = len(items)
    ncols = 1 if k == 1 else 3 if k == 3 else 2
    nrows = ceil(k / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    legend_handles = legend_labels = None

    for idx, (ax, item) in enumerate(zip(axes, items)):
        pre_col, post_col = f"S0_{item}", f"S1_{item}"
        if pre_col not in df.columns or post_col not in df.columns:
            raise KeyError(f"Missing Likert pair: {pre_col}, {post_col}")
        pre, post = paired_likert(df, pre_col, post_col)
        if not len(pre):
            raise ValueError(f"No valid paired Likert responses for {item}.")

        mean_pre = float(np.mean(pre))
        mean_post = float(np.mean(post))
        ax.hist(pre, bins=LIKERT_BINS, color=PURPLE, alpha=0.45, label="S1 (pre)")
        ax.hist(post, bins=LIKERT_BINS, color=TEAL, alpha=0.45, label="S2 (post)")
        ax.axvline(mean_pre, linestyle="--", linewidth=1.8, label="S1 mean", color=PURPLE)
        ax.axvline(mean_post, linestyle=":", linewidth=1.8, label="S2 mean", color=TEAL)
        ax.set_title(labels.get(item, item), fontsize=15)
        ax.set_xticks(LIKERT_VALUES)
        ax.tick_params(axis="both", labelsize=15)
        ax.set_xlim(0.5, 6.5)
        if idx % ncols == 0:
            ax.set_ylabel("# Respondents", fontsize=15)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Likert response", fontsize=15)
        else:
            ax.tick_params(labelbottom=False)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes[len(items):]:
        ax.axis("off")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=15,
        )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_deservingness_likert(
    df: pd.DataFrame,
    columns=DESERVINGNESS_COLUMNS,
    title_map=DESERVINGNESS_TITLES,
):
    """Plot deservingness Likert panel for specified deservingness columns."""
    columns = list(columns)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing deservingness columns: {missing}")

    ncols = 3
    nrows = ceil(len(columns) / ncols)
    fontsize = 17
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.0 * ncols, 3.4 * nrows),
        squeeze=False,
        sharey=True,
    )

    counts_by_col = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        values = values[values.isin(LIKERT_VALUES)].astype(int)
        counts_by_col[column] = values.value_counts().reindex(LIKERT_VALUES, fill_value=0).sort_index()

    global_max = max((counts.max() for counts in counts_by_col.values()), default=0)
    y_max = global_max * 1.12 if global_max > 0 else 1

    for idx, column in enumerate(columns):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        counts = counts_by_col[column]
        n_answered = int(counts.sum())
        title = f"{title_map.get(column, column)}\nN = {n_answered}"
        ax.bar(
            counts.index.astype(str),
            counts.values,
            width=0.75,
            edgecolor="black",
            color=LIGHT_PURPLE,
        )
        ax.set_title(
            "\n".join(textwrap.wrap(title, width=38, break_long_words=False)),
            fontsize=fontsize,
        )
        ax.set_xlabel("Likert value", fontsize=fontsize)
        ax.set_ylabel("# Respondents", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.set_xticks(range(len(LIKERT_VALUES)))
        ax.set_xticklabels([str(value) for value in LIKERT_VALUES], fontsize=fontsize)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.25)

    for idx in range(len(columns), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    return fig

def plot_likert_shifts(df: pd.DataFrame):
    items = [
        "sensitivity_nh_1",
        "sensitivity_nh_2",
        "sensitivity_nh_3",
        "finan_vulnerability_1",
    ]
    titles = {
        "sensitivity_nh_1": "Personal Natural hazard vulnerability",
        "sensitivity_nh_2": "Natural hazard vulnerability of family",
        "sensitivity_nh_3": "Natural hazard vulnerability of municipality",
        "finan_vulnerability_1": "Likert: Financial vulnerability",
    }

    delta_levels = np.arange(-5, 6)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4), sharey=True)
    axes = axes.ravel()

    for idx, (ax, item) in enumerate(zip(axes, items)):
        pre_col, post_col = f"S0_{item}", f"S1_{item}"
        if pre_col not in df or post_col not in df:
            raise KeyError(f"Missing Likert pair: {pre_col}, {post_col}")

        pre, post = paired_likert(df, pre_col, post_col)
        if not len(pre):
            raise ValueError(f"No valid paired Likert responses for {item}.")

        delta = post - pre
        counts = pd.Series(delta).value_counts().reindex(delta_levels, fill_value=0) / len(delta)
        mean = float(np.mean(delta))

        ax.bar(counts.index, counts.values, color=BROWN, alpha=0.5)
        ax.axvline(mean, color=PURPLE, linewidth=2)
        ax.set_title(titles[item], fontsize=14)
        ax.set_ylim(0, 0.55)
        ax.tick_params(labelsize=14)

        if idx % 2 == 0:
            ax.set_ylabel("Share [-]", fontsize=14)
        if idx // 2 == 1:
            ax.set_xlabel("Δ Likert = Post − Pre", fontsize=14)
        else:
            ax.tick_params(labelbottom=False)

        x_offset = -0.08 if mean > delta_levels.max() - 0.5 else 0.08
        ha = "right" if x_offset < 0 else "left"
        ax.text(
            mean + x_offset, 0.55 * 0.96, f"mean = {mean:.2f}",
            color=PURPLE, ha=ha, va="top", fontsize=14,
        )

    fig.tight_layout()
    return fig