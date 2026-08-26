from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

def set_thesis_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "figure.dpi": 150,
            "savefig.dpi": 800,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    
def save_figure(fig, output) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def posterior_interval(draws, hdi_prob: float = 0.94):
    values = np.asarray(draws).reshape(-1)
    hdi = az.hdi(values, hdi_prob=hdi_prob)
    return values.mean(), hdi[0], hdi[1]


def plot_intervals(ax, draws_by_label: dict, xlabel: str = "") -> None:
    labels = list(draws_by_label)
    y = np.arange(len(labels))

    stats = [posterior_interval(draws_by_label[label]) for label in labels]
    means = np.array([x[0] for x in stats])
    lower = np.array([x[1] for x in stats])
    upper = np.array([x[2] for x in stats])

    ax.errorbar(
        means,
        y,
        xerr=[means - lower, upper - means],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()