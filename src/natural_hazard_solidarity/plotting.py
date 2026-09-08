from pathlib import Path
import textwrap

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ATTRIBUTE_COLORS = {
    "costs": "#542788",
    "exemptions": "#01665e",
    "benefits": "#b35806",
}

PRETTY_ATTRIBUTES = {
    "costs": "Who should bear the costs?",
    "exemptions": "Should there be people exempted from the costs?",
    "benefits": "Which municipalities should be particularly protected?",
}


def set_thesis_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 800,
    })


def set_hypothesis_style() -> None:
    """Style from the old hypothesis plotting helper."""
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "font.size": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "savefig.dpi": 800,
    })
    


def posterior_interval(draws, hdi_prob=0.89, center="mean"):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]

    if draws.size == 0:
        return np.nan, np.nan, np.nan

    hdi = az.hdi(draws, hdi_prob=hdi_prob)
    middle = draws.mean() if center == "mean" else np.median(draws)

    return float(middle), float(hdi[0]), float(hdi[1])


def _draw_interval(ax, draws, y, color, alpha=1.0, lw=2.4, marker_size=90, cap=0.08, zorder=3):
    mid, lo, hi = posterior_interval(draws)
    if not np.isfinite(mid):
        return

    ax.hlines(y, lo, hi, color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.vlines([lo, hi], y - cap, y + cap, color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.scatter([mid], [y], s=marker_size, color=color, alpha=alpha, zorder=zorder + 1)


def _style_panel_axis(ax, xlim, show_ylabels=True):
    ax.set_xlim(*xlim)
    ax.axvline(0, color="#bdbdbd", linestyle="--", linewidth=1.0, zorder=0)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#b3b3b3")

    if not show_ylabels:
        ax.tick_params(axis="y", left=False, labelleft=False)
    else:
        ax.tick_params(axis="y", length=0)


def save_figure(fig, output) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(fig)


def wrap_label(text, width=50):
    return "\n".join(textwrap.wrap(str(text), width=width))


def possible_prefixes(attr_name: str):
    prefixes = [f"{attr_name}_"]
    if attr_name.endswith("s"):
        prefixes.append(f"{attr_name[:-1]}_")
    return prefixes


def find_posterior_name(attr_name: str, level_label: str, available_names):
    for prefix in possible_prefixes(attr_name):
        candidate = f"{prefix}{level_label}"
        if candidate in available_names:
            return candidate
    return None


def extract_draws_1d(da_1d):
    return np.asarray(da_1d).reshape(-1)


def get_level_draws_from_effectcoding(
    da, attr, level_label, levels_by_attr, baselines
):
    available_names = list(da.coords["level"].values)
    baseline_label = baselines[attr]
    name = find_posterior_name(attr, level_label, available_names)

    if name is not None:
        return extract_draws_1d(da.sel(level=name))

    if level_label != baseline_label:
        raise ValueError(
            f"Could not find posterior level '{attr}_{level_label}'. "
            f"Available levels: {available_names}"
        )

    nonbaseline_draws = []
    for level in levels_by_attr[attr]:
        if level == baseline_label:
            continue
        name = find_posterior_name(attr, level, available_names)
        if name is not None:
            nonbaseline_draws.append(extract_draws_1d(da.sel(level=name)))

    if not nonbaseline_draws:
        raise ValueError(f"No non-baseline coefficients found for '{attr}'.")

    return -np.sum(np.column_stack(nonbaseline_draws), axis=1)


def summarize_draws(draws, hdi_prob=0.89):
    mean, low, high = posterior_interval(draws, hdi_prob=hdi_prob)
    return {"mean": mean, "hdi_low": low, "hdi_high": high}


def build_effectcoded_summary_from_da(
    da,
    attr_order,
    levels_by_attr,
    baselines,
    kind_label,
    hdi_prob=0.89,
):
    rows = []
    draws_map = {}

    for attr in attr_order:
        for level in levels_by_attr[attr]:
            draws = get_level_draws_from_effectcoding(
                da=da,
                attr=attr,
                level_label=level,
                levels_by_attr=levels_by_attr,
                baselines=baselines,
            )
            summary = summarize_draws(draws, hdi_prob=hdi_prob)
            rows.append({
                "attribute": attr,
                "level": level,
                "kind": kind_label,
                **summary,
            })
            draws_map[(kind_label, attr, level)] = draws

    return pd.DataFrame(rows), draws_map


def add_row_titles_left_of_labels(
    fig,
    axes,
    attr_order,
    pretty_attr,
    row_header_fs=14,
    pad_pts=5,
    title_color_map=None,
):
    """Place row headers at the left edge of the y tick labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for row, attr in enumerate(attr_order):
        ax = axes[row, 0]
        labels = [label for label in ax.get_yticklabels() if label.get_text()]

        if labels:
            left_px = min(
                label.get_window_extent(renderer=renderer).x0
                for label in labels
            )
        else:
            left_px = ax.get_window_extent(renderer=renderer).x0

        top_px = ax.get_window_extent(renderer=renderer).y1
        x_fig, y_fig = inv.transform((left_px, top_px))
        pad_fig = (pad_pts / 72.0) / fig.get_size_inches()[1]

        color = "black"
        if title_color_map is not None:
            color = title_color_map.get(attr, "black")

        fig.text(
            x_fig,
            y_fig + pad_fig,
            pretty_attr.get(attr, attr),
            ha="left",
            va="bottom",
            fontsize=row_header_fs,
            fontweight="bold",
            color=color,
        )


def _draw_density(
    ax,
    draws,
    y,
    color,
    height,
    alpha,
    xlim,
    line_alpha=1.0,
    line_width=0.7,
):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]

    if draws.size < 3 or np.ptp(draws) == 0:
        return

    lo = max(float(np.min(draws)), xlim[0])
    hi = min(float(np.max(draws)), xlim[1])

    if hi <= lo:
        return

    xs = np.linspace(lo, hi, 250)
    density = gaussian_kde(draws)(xs)

    if density.max() <= 0:
        return

    density = density / density.max() * height

    # y-axis is inverted in the forest plots:
    # y - density therefore appears visually ABOVE the data row.
    density_y = y - density

    # One-sided posterior density.
    ax.fill_between(xs, y, density_y, color=color, alpha=alpha, linewidth=0, zorder=1,)

    # Dark outline of the posterior density.
    ax.plot(xs, density_y, color=color, alpha=line_alpha, linewidth=line_width, zorder=2,)
    
def _draw_interval_with_caps(
    ax,
    mean,
    low,
    high,
    y,
    color,
    alpha=1.0,
    linewidth=1.5,
    cap_height=0.10,
    point_size=62,
):
    ax.hlines(y, low, high, color=color, alpha=alpha, linewidth=linewidth, zorder=3)
    ax.vlines([low, high], y - cap_height, y + cap_height, color=color, alpha=alpha, linewidth=linewidth, zorder=3)
    ax.scatter(mean, y, s=point_size, color=color, alpha=alpha, zorder=4)

def plot_forest_panels(
    plot_df,
    attr_order,
    levels_by_attr,
    panels,
    subplot_color_map,
    draws_map=None,
    show_density=False,
    density_height=0.42,
    density_alpha=0.18,
    point_size=62,
    line_width=2.2,
    fig_width=11,
    left_margin=0.45,
    per_level_height=0.40,
    level_wrap_width=50,
    vconnect=False,
    row_titles=None,
    row_title_fs=14,
    row_title_colors=None,
):
    nrows = len(attr_order)
    ncols = len(panels)

    height_ratios = [
        max(len(levels_by_attr[attr]), 1)
        for attr in attr_order
    ]

    fig_height = max(
        4.2,
        sum(height_ratios) * per_level_height + 1.4,
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )

    for row, attr in enumerate(attr_order):
        levels = levels_by_attr[attr]
        y_positions = np.arange(len(levels))

        for col, panel in enumerate(panels):
            ax = axes[row, col]

            kinds = panel["kinds"]
            alphas = panel.get(
                "alphas",
                [1.0] * len(kinds),
            )

            if len(alphas) < len(kinds):
                alphas = list(alphas) + [
                    1.0
                ] * (len(kinds) - len(alphas))

            xlim = panel["xlim"]

            # Allows H1 left = no density,
            # H1 right = density,
            # H2/H3 = density.
            panel_show_density = panel.get(
                "show_density",
                show_density,
            )

            if panel.get("zero", 0):
                ax.axvline(
                    0,
                    color="0.55",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=0,
                )

            means_by_kind = []

            for kind_index, kind in enumerate(kinds):
                means = []

                alpha = alphas[kind_index]

                color = subplot_color_map.get(
                    (attr, kind),
                    ATTRIBUTE_COLORS[attr],
                )

                # Pre/post are slightly vertically separated.
                offset = 0.0
                if len(kinds) > 1:
                    offset = (
                        kind_index
                        - (len(kinds) - 1) / 2
                    ) * 0.16

                for y, level in zip(
                    y_positions,
                    levels,
                ):
                    row_df = plot_df[
                        (plot_df["attribute"] == attr)
                        & (plot_df["level"] == level)
                        & (plot_df["kind"] == kind)
                    ]

                    if row_df.empty:
                        means.append(np.nan)
                        continue

                    result = row_df.iloc[0]
                    yy = y + offset

                    draws = (
                        None
                        if draws_map is None
                        else draws_map.get(
                            (kind, attr, level)
                        )
                    )

                    # Posterior distribution where requested.
                    if (
                        panel_show_density
                        and draws is not None
                    ):
                        _draw_density(
                            ax,
                            draws,
                            yy,
                            color=color,
                            height=density_height,
                            alpha=density_alpha * alpha,
                            xlim=xlim,
                        )

                    # HDI line + end caps + posterior mean.
                    _draw_interval_with_caps(
                        ax,
                        mean=result["mean"],
                        low=result["hdi_low"],
                        high=result["hdi_high"],
                        y=yy,
                        color=color,
                        alpha=alpha,
                        linewidth=line_width,
                        cap_height=0.08,
                        point_size=point_size,
                    )

                    means.append(result["mean"])

                means_by_kind.append(means)

            if vconnect and len(means_by_kind) > 1:
                for level_index, y in enumerate(
                    y_positions
                ):
                    xs = [
                        means[level_index]
                        for means in means_by_kind
                    ]

                    if np.all(np.isfinite(xs)):
                        ax.plot(
                            xs,
                            [y] * len(xs),
                            color="0.6",
                            linewidth=0.8,
                            zorder=1,
                        )

            ax.set_xlim(*xlim)

            ax.set_ylim(
                len(levels) - 0.5,
                -0.5,
            )

            ax.set_yticks(y_positions)

            if col == 0:
                ax.set_yticklabels(
                    [
                        wrap_label(
                            level,
                            level_wrap_width,
                        )
                        for level in levels
                    ]
                )
            else:
                ax.set_yticklabels([])

            if row == 0 and panel.get("title"):
                ax.set_title(
                    panel["title"],
                    pad=12,
                )

            # Vertical grid only.
            ax.grid(
                axis="x",
                color="0.88",
                linewidth=0.8,
            )

            ax.set_axisbelow(True)

            ax.tick_params(
                axis="y",
                length=0,
            )

            # Full frame around every subplot.
            for side in (
                "top",
                "right",
                "left",
                "bottom",
            ):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color("0.72")
                ax.spines[side].set_linewidth(0.8)

            # Only bottom row gets x labels.
            if row < nrows - 1:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )
                ax.set_xlabel("")
            else:
                ax.set_xlabel(
                    panel.get("xlabel", "")
                )

    fig.subplots_adjust(
        left=left_margin,
        right=0.98,
        top=0.96,
        bottom=0.15,
        hspace=0.18,
        wspace=0.10,
    )

    add_row_titles_left_of_labels(
        fig,
        axes,
        attr_order,
        row_titles or PRETTY_ATTRIBUTES,
        row_header_fs=row_title_fs,
        pad_pts=5,
        title_color_map=row_title_colors,
    )

    return fig, axes