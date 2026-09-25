"""Posterior diagnostic figures"""
from __future__ import annotations

from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde

from .plotting import LEVEL_LABELS, LEVEL_ORDER
from .posterior import check_model_coding

def _trace_groups(conjoint_config):
    """Return non-baseline level coordinates grouped"""
    titles = {
        "costs": "Costs",
        "exemptions": "Cost exemptions",
        "benefits": "Benefits",
    }
    groups = {}
    pretty = {}

    for attribute in ("costs", "exemptions", "benefits"):
        spec = conjoint_config["attributes"][attribute]
        configured = list(spec["levels"])
        ordered = [
            level
            for level in LEVEL_ORDER.get(attribute, configured)
            if level in configured
        ]
        levels = [level for level in ordered if level != spec["baseline"]]
        coordinates = [f"{attribute}_{level}" for level in levels]
        groups[titles[attribute]] = coordinates

        for level, coordinate in zip(levels, coordinates):
            label = LEVEL_LABELS.get(level, level).replace("CO2", "CO$_2$")
            if level == "Municipalities most affected by natural hazards even if they are economically declining":
                label = "Municipalities most affected by natural hazards"
            elif level == "Municipalities in which people have lived in for many years should be protected at all costs":
                label = "Municipalities with long-standing residents"
            pretty[coordinate] = label

    return groups, pretty


def _kde(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.ptp(values) == 0:
        return None, None
    lo, hi = float(values.min()), float(values.max())
    pad = 0.05 * max(hi - lo, 1e-6)
    xs = np.linspace(lo - pad, hi + pad, 300)
    return xs, gaussian_kde(values)(xs)


def _pretty_variable_label(variable):
    labels = {
        "partworth_mean": r"Partworth utility $\beta_k$",
        "shift_mean": r"Average utility shift $\delta_k$",
    }
    return labels.get(variable, variable.replace("_", " "))

def plot_partworth_trace(idata, conjoint_config, variable="shift_mean"):
    """Posterior densities and chain traces for conjoint-level parameters.

    Colours identify conjoint levels; line styles identify MCMC chains.
    Densities are shown separately for every chain without filled areas.
    """
    check_model_coding(idata, conjoint_config)
    if variable not in idata.posterior:
        raise KeyError(f"Posterior variable not found: {variable}")

    posterior = idata.posterior[variable]
    if "level" not in posterior.dims:
        raise ValueError(f"Posterior variable {variable!r} has no 'level' dimension.")

    level_groups, pretty_titles = _trace_groups(conjoint_config)
    available = set(map(str, posterior["level"].values))
    cleaned = {
        group: [level for level in levels if level in available]
        for group, levels in level_groups.items()
    }
    cleaned = {group: levels for group, levels in cleaned.items() if levels}
    if not cleaned:
        raise ValueError(f"No configured conjoint levels were found in {variable}.")

    chains = list(posterior.chain.values)
    draws = np.arange(posterior.sizes["draw"])
    linestyles = ["-", "--", ":", "-."]
    chain_styles = {chain: linestyles[i % len(linestyles)] for i, chain in enumerate(chains)}

    fig, axes = plt.subplots(len(cleaned), 2, figsize=(13.5, 3.65 * len(cleaned)), squeeze=False)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    variable_label = _pretty_variable_label(variable)

    for row, (group_name, group_levels) in enumerate(cleaned.items()):
        ax_post, ax_trace = axes[row]
        level_colors = {level: palette[i % len(palette)] for i, level in enumerate(group_levels)}

        for level in group_levels:
            color = level_colors[level]

            for chain in chains:
                values = np.asarray(posterior.sel(level=level, chain=chain).values, dtype=float)

                # Posterior density: one line per chain, no fill.
                xs, ys = _kde(values)
                if xs is not None:
                    ax_post.plot(
                        xs, ys, color=color, linestyle=chain_styles[chain],
                        linewidth=1.5,
                    )

                # Trace: same colour/line-style convention.
                ax_trace.plot(
                    draws, values, color=color, linestyle=chain_styles[chain],
                    linewidth=0.75, alpha=0.72,
                )

        ax_post.axvline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
        ax_trace.axhline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
        ax_post.grid(axis="x", color="0.9", linewidth=0.6)
        ax_trace.grid(color="0.9", linewidth=0.55, alpha=0.7)

        ax_post.set_xlabel(variable_label)
        ax_post.set_ylabel("Posterior density")
        ax_trace.set_xlabel("Posterior draw")
        ax_trace.set_ylabel(variable_label)

        ax_post.set_title(f"{group_name}: Posterior distributions", loc="left", fontweight="bold")
        ax_trace.set_title(f"{group_name}: Trace plots", loc="left", fontweight="bold")

        level_handles = [
            Line2D(
                [0], [0], color=level_colors[level], lw=2.2,
                label="\n".join(wrap(pretty_titles.get(level, level), width=52)),
            )
            for level in group_levels
        ]

        chain_handles = [
            Line2D(
                [0], [0], color="black", linestyle=chain_styles[chain], lw=1.4,
                label=f"Chain {i + 1}",
            )
            for i, chain in enumerate(chains)
        ]

        level_legend = ax_post.legend(
            handles=level_handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
            ncol=2 if len(level_handles) > 2 else max(1, len(level_handles)),
            frameon=False, handlelength=2.6, columnspacing=1.1,
        )
        ax_post.add_artist(level_legend)

        ax_trace.legend(
            handles=chain_handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
            ncol=min(4, max(1, len(chain_handles))),
            frameon=False, handlelength=2.4, columnspacing=1.4,
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.08, hspace=0.72, wspace=0.18)

    return fig