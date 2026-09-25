"""General plotting functions"""
from pathlib import Path
import textwrap
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import gaussian_kde

PURPLE = "#542788"
LIGHT_PURPLE = "#998ec3"
VERY_LIGHT_PURPLE = "#d2c9f2"
TEAL = "#01665e"
LIGHT_TEAL = "#5ab4ac"
BROWN = "#b35806"
LIGHT_BROWN = "#f1a340"
NEUTRAL = "#f7f7f7"
DARK_GREY = "#1F2022"

ATTRIBUTE_COLORS = {"costs": PURPLE, "exemptions": TEAL, "benefits": BROWN,}

TEAL_BROWN = LinearSegmentedColormap.from_list("teal_brown", [TEAL, LIGHT_TEAL, NEUTRAL, LIGHT_BROWN, BROWN])
BROWN_PURPLE = LinearSegmentedColormap.from_list("brown_purple", [BROWN, LIGHT_BROWN, NEUTRAL, LIGHT_PURPLE, PURPLE])
PURPLE_BROWN = BROWN_PURPLE.reversed(name="purple_brown")

# Shared presentation defaults. Scientific choices belong in config.yaml;
# colors/colormaps belong here so all figures use one thesis style.
ROBUSTNESS_COLORS = (DARK_GREY, TEAL, PURPLE, BROWN)
ROBUSTNESS_ALPHAS = (0.90, 0.60, 0.60, 0.60)
COUNT_CMAP = plt.get_cmap("Purples")
MAP_CMAP = PURPLE_BROWN

ATTRIBUTE_ORDER = ("costs", "exemptions", "benefits")
PRETTY_ATTRIBUTES = {
    "costs": "Costs: Who should bear the costs?",
    "exemptions": "Cost exemptions: Should there be people exempted from the costs?",
    "benefits": "Benefits: Which municipalities should be particularly protected?",
}

LEVEL_ORDER = {
    "costs": ["All people pay the same amount", "Companies pay proportionally to their CO2 emissions",
              "People pay proportionally to their CO2 emissions", "People & companies being protected by protective measures",
              "People pay proportionally to their income"],
    "exemptions": ["No groups exempted from costs", "Low- and middle-income earners exempted from costs",
                   "Low-income earners exempted from costs"],
    "benefits": ["Equal protection levels for all municipalities", "Culturally valuable municipalities e.g. with historic buildings",
                 "Economically prosperous municipalities", "Municipalities most affected by natural hazards even if they are economically declining",
                 "Municipalities in which people have lived in for many years should be protected at all costs"],
}

LEVEL_LABELS = {
    "People & companies being protected by protective measures": "People and companies protected by protection measures",
    "Equal protection levels for all municipalities": "Equal protection levels for all",
    "No groups exempted from costs": "No groups are exempted from costs",
    "Low- and middle-income earners exempted from costs": "Low- and middle-income earners exempted",
    "Low-income earners exempted from costs": "Low-income earners exempted",
    "Culturally valuable municipalities e.g. with historic buildings": "Culturally valuable municipalities",
    "Municipalities most affected by natural hazards even if they are economically declining": "Municipalities most affected by natural hazards even when economically declining",
    "Municipalities in which people have lived in for many years should be protected at all costs": "Municipalities in which people have lived for many years",
}


def set_thesis_style():
    mpl.rcParams.update({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 11,
                         "axes.titlesize": 12, "legend.fontsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "figure.dpi": 120, "savefig.dpi": 800})


def save_figure(fig, output, dpi=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(output, **kwargs)
    plt.close(fig)


def wrap_label(text, width=50):
    label = LEVEL_LABELS.get(str(text), str(text)).replace("CO2", "CO$_2$")
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _draw_density(ax, draws, y, color, height, alpha, xlim, line_alpha=1.0, line_width=0.7):
    values = np.asarray(draws, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 3 or np.ptp(values) == 0:
        return
    lo, hi = max(values.min(), xlim[0]), min(values.max(), xlim[1])
    if hi <= lo:
        return
    xs = np.linspace(lo, hi, 250)
    density = gaussian_kde(values)(xs)
    density = density / density.max() * height
    curve = y - density  # inverted y axis: upwards only
    ax.fill_between(xs, y, curve, color=color, alpha=alpha, linewidth=0, zorder=1)
    ax.plot(xs, curve, color=color, alpha=line_alpha, linewidth=line_width, zorder=2)


def _draw_interval_with_caps(ax, mean, low, high, y, color, alpha=1.0, linewidth=1.7, cap_height=0.08, point_size=62):
    # Draw absolute HDI endpoints: a posterior mean need not lie inside an HDI.
    ax.hlines(y, low, high, color=color, alpha=alpha, linewidth=linewidth, zorder=3)
    ax.vlines([low, high], y-cap_height, y+cap_height, color=color, alpha=alpha, linewidth=linewidth, zorder=3)
    ax.scatter(mean, y, s=point_size, color=color, alpha=alpha, zorder=4)


def add_row_titles_left_of_labels(fig, axes, attr_order, pretty_attr, row_header_fs=12, pad_pts=5, title_color_map=None):
    """Keep headers attached to axes; subsequent layout changes cannot detach them."""
    for row, attr in enumerate(attr_order):
        ax = axes[row, 0]
        color = (title_color_map or {}).get(attr, "black")
        ax.annotate(pretty_attr.get(attr, attr), xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, pad_pts), textcoords="offset points", ha="left", va="bottom",
                    fontsize=row_header_fs, fontweight="bold", color=color, annotation_clip=False)


def plot_forest_panels(plot_df, attr_order, levels_by_attr, panels, subplot_color_map,
                       draws_map=None, show_density=False, density_height=0.40, density_alpha=0.20,
                       point_size=62, line_width=1.5, fig_width=11.4, left_margin=0.44,
                       per_level_height=0.42, level_wrap_width=50, vconnect=False,
                       row_titles=None, row_title_fs=12, row_title_colors=None, bottom_margin=0.16):
    """Stable plot_df/panels API. Never introduce a second panel_specs API here."""
    nrows, ncols = len(attr_order), len(panels)
    heights = [len(levels_by_attr[a]) for a in attr_order]
    if not nrows or not ncols or min(heights) == 0:
        raise ValueError("Forest plot requires nonempty attributes, levels and panels.")
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(fig_width, sum(heights)*per_level_height+1.7),
                             gridspec_kw={"height_ratios": heights})
    for row, attr in enumerate(attr_order):
        levels, positions = levels_by_attr[attr], np.arange(heights[row])
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            kinds = panel["kinds"]
            alphas = panel.get("alphas", [1.0]*len(kinds))
            if len(alphas) != len(kinds):
                raise ValueError("One alpha is required for every panel kind.")
            xlim = panel["xlim"]
            if panel.get("zero", True):
                ax.axvline(0, color="0.55", linestyle="--", linewidth=0.8, zorder=0)
            all_means = []
            for k, kind in enumerate(kinds):
                means = []
                color, alpha = subplot_color_map[(attr, kind)], alphas[k]
                offset = (k - (len(kinds)-1)/2)*panel.get("offset", 0.16)
                for y, level in zip(positions, levels):
                    records = plot_df.loc[(plot_df.attribute == attr) & (plot_df.level == level) & (plot_df.kind == kind)]
                    if len(records) != 1:
                        raise ValueError(f"Expected one summary for {(kind, attr, level)}, found {len(records)}.")
                    summary = records.iloc[0]
                    yy = y + offset
                    draws = None if draws_map is None else draws_map.get((kind, attr, level))
                    if panel.get("show_density", show_density) and draws is not None:
                        _draw_density(ax, draws, yy, color, density_height, density_alpha*alpha, xlim)
                    _draw_interval_with_caps(ax, summary["mean"], summary.hdi_low, summary.hdi_high, yy,
                                             color, alpha, line_width, 0.08, point_size)
                    means.append(summary["mean"])
                all_means.append(means)
            if vconnect and len(kinds) > 1:
                for index, y in enumerate(positions):
                    ax.plot([m[index] for m in all_means], [y]*len(kinds), color="0.7", linewidth=0.7, zorder=0)
            ax.set(xlim=xlim, ylim=(len(levels)-0.5, -0.5), yticks=positions)
            ax.set_yticklabels([wrap_label(x, level_wrap_width) for x in levels] if col == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color="0.88", linewidth=0.7); ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color("0.72"); spine.set_linewidth(0.8)
            if row < nrows-1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(panel.get("xlabel", panel.get("title", "")), labelpad=9)
    fig.subplots_adjust(left=left_margin, right=0.98, top=0.94, bottom=bottom_margin, hspace=0.30, wspace=0.14)
    add_row_titles_left_of_labels(fig, axes, attr_order, row_titles or PRETTY_ATTRIBUTES,
                                 row_header_fs=row_title_fs, title_color_map=row_title_colors)
    return fig, axes

set_thesis_style()