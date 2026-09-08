from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


SNAKEMAKE = globals().get("snakemake")
if SNAKEMAKE is not None:
    REPO_ROOT = Path(SNAKEMAKE.scriptdir).parents[1]
else:
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from natural_hazard_solidarity.plotting import (
    ATTRIBUTE_COLORS,
    build_effectcoded_summary_from_da,
    get_level_draws_from_effectcoding,
    plot_forest_panels,
    save_figure,
    set_hypothesis_style,
)


set_hypothesis_style()

ATTR_ORDER = ["costs", "exemptions", "benefits"]

# This is the order used in the old analyze_results.ipynb figures.
LEVEL_ORDER = {
    "costs": [
        "All people pay the same amount",
        "Companies pay proportionally to their CO2 emissions",
        "People pay proportionally to their CO2 emissions",
        "People & companies being protected by protective measures",
        "People pay proportionally to their income",
    ],
    "exemptions": [
        "No groups exempted from costs",
        "Low- and middle-income earners exempted from costs",
        "Low-income earners exempted from costs",
    ],
    "benefits": [
        "Equal protection levels for all municipalities",
        "Culturally valuable municipalities e.g. with historic buildings",
        "Economically prosperous municipalities",
        "Municipalities most affected by natural hazards even if they are economically declining",
        "Municipalities in which people have lived in for many years should be protected at all costs",
    ],
}

BLUE_BROWN = LinearSegmentedColormap.from_list(
    "blue_brown",
    ["#01665e", "#5ab4ac", "#f7f7f7", "#f1a340", "#b35806"],
)

PRETTY_ATTR = {
    "costs": "Who should bear the costs?",
    "exemptions": "Should there be people exempted from the costs?",
    "benefits": "Which municipalities should be particularly protected?",
}


def _baselines(conjoint_config):
    return {
        attr: conjoint_config["attributes"][attr]["baseline"]
        for attr in ATTR_ORDER
    }


def _levels_from_config(conjoint_config):
    return {
        attr: list(conjoint_config["attributes"][attr]["levels"])
        for attr in ATTR_ORDER
    }


def _validate_old_level_order(levels_by_attr):
    for attr in ATTR_ORDER:
        if set(LEVEL_ORDER[attr]) != set(levels_by_attr[attr]):
            raise ValueError(
                f"Configured levels for '{attr}' do not match the old plotting order. "
                "Update LEVEL_ORDER in plot_hypotheses.py if the conjoint design changed."
            )


def _subplot_color_map(kinds):
    return {
        (attr, kind): ATTRIBUTE_COLORS[attr]
        for attr in ATTR_ORDER
        for kind in kinds
    }


def _padded_xlim(plot_df, kinds, pad_fraction=0.05):
    mask = plot_df["kind"].isin(kinds)
    low = plot_df.loc[mask, "hdi_low"].min()
    high = plot_df.loc[mask, "hdi_high"].max()
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Could not determine plot limits from posterior summaries.")
    if high == low:
        return float(low - 0.1), float(high + 0.1)
    pad = pad_fraction * (high - low)
    return float(low - pad), float(high + pad)

def plot_h1(idata, output, conjoint_config):
    posterior = idata.posterior
    baselines = _baselines(conjoint_config)
    levels_by_attr = _levels_from_config(conjoint_config)

    pre_total_da = posterior["partworth_mean"]
    shift_total_da = posterior["shift_mean"]
    post_total_da = pre_total_da + shift_total_da

    pre_df, pre_draws = build_effectcoded_summary_from_da(
        da=pre_total_da,
        attr_order=ATTR_ORDER,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        kind_label="pre_total",
    )

    post_df, post_draws = build_effectcoded_summary_from_da(
        da=post_total_da,
        attr_order=ATTR_ORDER,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        kind_label="post_total",
    )

    shift_df, shift_draws = build_effectcoded_summary_from_da(
        da=shift_total_da,
        attr_order=ATTR_ORDER,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        kind_label="shift_total",
    )

    plot_df = pd.concat(
        [pre_df, post_df, shift_df],
        ignore_index=True,
    )

    draws_map = {
        **pre_draws,
        **post_draws,
        **shift_draws,
    }

    left_xlim = _padded_xlim(
        plot_df,
        ["pre_total", "post_total"],
        pad_fraction=0.05,
    )
    
    panels = [
        {
            "kinds": ["pre_total", "post_total"],
            "alphas": [0.35, 1.0],
            "title": "",
            "xlabel": (
                "Pre average partworth utility $\\beta_k$   /   "
                "Post average partworth utility $\\beta_k^{eff}$"
            ),
            "xlim": left_xlim,
            "zero": 1,
            "show_density": False,
        },
        {
            "kinds": ["shift_total"],
            "alphas": [1.0],
            "title": "",
            "xlabel": "Average utility shift\n$\\delta_k$",
            "xlim": left_xlim,
            "zero": 1,
            "show_density": True,
        },
    ]

    fig, _ = plot_forest_panels(
        plot_df=plot_df,
        attr_order=ATTR_ORDER,
        levels_by_attr=LEVEL_ORDER,
        panels=panels,
        subplot_color_map=_subplot_color_map(
            ["pre_total", "post_total", "shift_total"]
        ),
        draws_map=draws_map,
        show_density=False,
        density_height=0.30,
        density_alpha=0.20,
        point_size=62,
        line_width=2.2,
        fig_width=11.4,
        left_margin=0.44,
        per_level_height=0.42,
        level_wrap_width=50,
        row_titles={
            "costs": "Costs: Who should bear the costs?",
            "exemptions": (
                "Cost exemptions: Should there be people "
                "exempted from the costs?"
            ),
            "benefits": (
                "Benefits: Which municipalities should be "
                "particularly protected?"
            ),
        },
        row_title_fs=15,
        row_title_colors={
            attr: "black"
            for attr in ATTR_ORDER
        },
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color="0.25",
            markerfacecolor="0.25",
            alpha=0.35,
            linewidth=2.2,
            markersize=7,
            label=r"Pre average partworth utility $\beta_k$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color="0.10",
            markerfacecolor="0.10",
            alpha=1.0,
            linewidth=2.2,
            markersize=7,
            label=r"Post average partworth utility $\beta_k^{eff}$",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        handlelength=1.2,
    )

    save_figure(fig, output)
    
def _build_h2(idata, conjoint_config):
    posterior = idata.posterior
    baselines = _baselines(conjoint_config)
    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_old_level_order(levels_by_attr)

    # Same direction convention as in the old notebook.
    pd_sign = -1.0

    gamma_pd_da = pd_sign * posterior["gamma_pd"]
    shift_psy_da = pd_sign * posterior["shift_psy"]

    pd_pre_df, pd_pre_draws = build_effectcoded_summary_from_da(
        da=gamma_pd_da,
        attr_order=ATTR_ORDER,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        kind_label="pd_pre",
    )

    pd_shift_df, pd_shift_draws = build_effectcoded_summary_from_da(
        da=shift_psy_da,
        attr_order=ATTR_ORDER,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        kind_label="pd_shift",
    )

    plot_df = pd.concat(
        [pd_pre_df, pd_shift_df],
        ignore_index=True,
    )

    draws_map = {
        **pd_pre_draws,
        **pd_shift_draws,
    }

    xlim = _padded_xlim(
        plot_df,
        ["pd_pre", "pd_shift"],
        pad_fraction=0.05,
    )

    return plot_df, draws_map, xlim

def plot_h2(idata, output, conjoint_config):
    plot_df, draws_map, xlim = _build_h2(
        idata,
        conjoint_config,
    )

    panels = [
        {
            "kinds": ["pd_pre"],
            "alphas": [1.0],
            "title": "",
            "xlabel": (
                "Pre-event utility association\n"
                "$\\gamma_{PD}$"
            ),
            "xlim": xlim,
            "zero": 1,
            "show_density": True,
        },
        {
            "kinds": ["pd_shift"],
            "alphas": [1.0],
            "title": "",
            "xlabel": (
                "Utility shift association\n"
                "$\\phi_{PD}$"
            ),
            "xlim": xlim,
            "zero": 1,
            "show_density": True,
        },
    ]

    fig, _ = plot_forest_panels(
        plot_df=plot_df,
        attr_order=ATTR_ORDER,
        levels_by_attr=LEVEL_ORDER,
        panels=panels,
        subplot_color_map=_subplot_color_map(
            ["pd_pre", "pd_shift"]
        ),
        draws_map=draws_map,

        # Panel settings above decide where densities appear.
        show_density=False,
        density_height=0.30,
        density_alpha=0.20,

        point_size=62,
        line_width=2.2,
        fig_width=11.4,
        left_margin=0.44,
        per_level_height=0.42,
        level_wrap_width=50,

        row_titles={
            "costs": "Costs: Who should bear the costs?",
            "exemptions": (
                "Cost exemptions: Should there be people "
                "exempted from the costs?"
            ),
            "benefits": (
                "Benefits: Which municipalities should be "
                "particularly protected?"
            ),
        },
        row_title_fs=15,
        row_title_colors={
            attr: "black"
            for attr in ATTR_ORDER
        },
    )

    save_figure(fig, output)

def plot_h3(idata, output, conjoint_config):
    posterior = idata.posterior
    baselines = _baselines(conjoint_config)
    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_old_level_order(levels_by_attr)

    if "shift_fin" not in posterior:
        raise ValueError(
            "H3 requires a model fitted with financial vulnerability."
        )

    shift_fin_da = posterior["shift_fin"]

    fin_shift_df, fin_shift_draws = build_effectcoded_summary_from_da(
        da=shift_fin_da,
        attr_order=ATTR_ORDER,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        kind_label="fin_shift",
    )

    # Keep the same x-axis scale as H2, as in the old notebook.
    _, _, h2_xlim = _build_h2(
        idata,
        conjoint_config,
    )

    panels = [
        {
            "kinds": ["fin_shift"],
            "alphas": [1.0],
            "title": "",
            "xlabel": (
                "Utility shift association\n"
                "$\\phi_{FV}$"
            ),
            "xlim": h2_xlim,
            "zero": 1,
            "show_density": True,
        },
    ]

    fig, _ = plot_forest_panels(
        plot_df=fin_shift_df,
        attr_order=ATTR_ORDER,
        levels_by_attr=LEVEL_ORDER,
        panels=panels,
        subplot_color_map=_subplot_color_map(
            ["fin_shift"]
        ),
        draws_map=fin_shift_draws,

        show_density=False,
        density_height=0.30,
        density_alpha=0.20,

        point_size=62,
        line_width=2.2,
        fig_width=6.5,
        left_margin=0.52,
        per_level_height=0.42,
        level_wrap_width=50,

        row_titles={
            "costs": "Costs: Who should bear the costs?",
            "exemptions": (
                "Cost exemptions: Should there be people "
                "exempted from the costs?"
            ),
            "benefits": (
                "Benefits: Which municipalities should be "
                "particularly protected?"
            ),
        },
        row_title_fs=15,
        row_title_colors={
            attr: "black"
            for attr in ATTR_ORDER
        },
    )

    save_figure(fig, output)

def make_axis_from_values(values, sign=1, n=80, pad_frac=0.03, use_quantiles=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    axis_values = sign * values
    if use_quantiles:
        lo, hi = np.quantile(axis_values, [0.055, 0.945])
    else:
        lo, hi = np.min(axis_values), np.max(axis_values)
    pad = pad_frac * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n)


def make_regular_ticks(lo, hi, step=0.5):
    start = np.ceil(lo / step) * step
    stop = np.floor(hi / step) * step
    ticks = np.arange(start, stop + step * 0.5, step)
    return np.round(ticks, 10)


def compute_level_surface(
    idata,
    attr,
    level_label,
    levels_by_attr,
    baselines,
    nhv_axis,
    fv_axis,
    pd_fixed=0.0,
    nhv_to_model_sign=1,
    fv_to_model_sign=1,
    quantity="post",
    hdi_prob=0.89,
):
    post = idata.posterior

    pw_mean_draws = get_level_draws_from_effectcoding(
        post["partworth_mean"], attr, level_label, levels_by_attr, baselines
    )
    gamma_nhv_draws = get_level_draws_from_effectcoding(
        post["gamma_nhv"], attr, level_label, levels_by_attr, baselines
    )
    gamma_fv_draws = get_level_draws_from_effectcoding(
        post["gamma_fv"], attr, level_label, levels_by_attr, baselines
    )
    gamma_pd_draws = get_level_draws_from_effectcoding(
        post["gamma_pd"], attr, level_label, levels_by_attr, baselines
    )

    shift_mean_draws = get_level_draws_from_effectcoding(
        post["shift_mean"], attr, level_label, levels_by_attr, baselines
    )
    shift_nh_draws = get_level_draws_from_effectcoding(
        post["shift_nh"], attr, level_label, levels_by_attr, baselines
    )
    shift_fin_draws = get_level_draws_from_effectcoding(
        post["shift_fin"], attr, level_label, levels_by_attr, baselines
    )
    shift_psy_draws = get_level_draws_from_effectcoding(
        post["shift_psy"], attr, level_label, levels_by_attr, baselines
    )

    z_mean = np.zeros((len(fv_axis), len(nhv_axis)))
    z_hdi_low = np.zeros_like(z_mean)
    z_hdi_high = np.zeros_like(z_mean)

    for i, fv_vuln in enumerate(fv_axis):
        fv_model = fv_to_model_sign * fv_vuln
        for j, nhv_vuln in enumerate(nhv_axis):
            nhv_model = nhv_to_model_sign * nhv_vuln

            pre_draws = (
                pw_mean_draws
                + gamma_nhv_draws * nhv_model
                + gamma_fv_draws * fv_model
                + gamma_pd_draws * pd_fixed
            )
            shift_draws = (
                shift_mean_draws
                + shift_nh_draws * nhv_model
                + shift_fin_draws * fv_model
                + shift_psy_draws * pd_fixed
            )

            if quantity == "post":
                draws = pre_draws + shift_draws
            elif quantity == "shift":
                draws = shift_draws
            elif quantity == "pre":
                draws = pre_draws
            else:
                raise ValueError("quantity must be 'pre', 'shift', or 'post'.")

            z_mean[i, j] = np.mean(draws)
            hdi = az.hdi(draws, hdi_prob=hdi_prob)
            z_hdi_low[i, j] = hdi[0]
            z_hdi_high[i, j] = hdi[1]

    return z_mean, z_hdi_low, z_hdi_high


def _wrap_title(text, width=38):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width=width))


def plot_multi_level_heatmaps_flat(
    idata,
    levels_h4,
    levels_by_attr,
    baselines,
    nhv_axis,
    fv_axis,
    quantity="shift",
    pd_fixed=0.0,
    nhv_to_model_sign=1,
    fv_to_model_sign=1,
):
    panels = [
        (attr, level)
        for attr, levels in levels_h4.items()
        for level in levels
    ]
    n_panels = len(panels)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.07, wspace=0.05)

    for ax, (attr, level) in zip(axes.flat, panels):
        z_mean, z_low, z_high = compute_level_surface(
            idata=idata,
            attr=attr,
            level_label=level,
            levels_by_attr=levels_by_attr,
            baselines=baselines,
            nhv_axis=nhv_axis,
            fv_axis=fv_axis,
            pd_fixed=pd_fixed,
            nhv_to_model_sign=nhv_to_model_sign,
            fv_to_model_sign=fv_to_model_sign,
            quantity=quantity,
        )

        vmin = float(np.min(z_low))
        vmax = float(np.max(z_high))
        image = ax.imshow(
            z_mean,
            origin="lower",
            aspect="auto",
            extent=[nhv_axis.min(), nhv_axis.max(), fv_axis.min(), fv_axis.max()],
            cmap=BLUE_BROWN,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(_wrap_title(level), fontsize=16)
        ax.set_xlabel("Natural hazard vulnerability", fontsize=15)
        ax.set_ylabel("Financial vulnerability", fontsize=15)
        ax.set_xticks(make_regular_ticks(nhv_axis.min(), nhv_axis.max(), step=0.5))
        ax.set_yticks(make_regular_ticks(fv_axis.min(), fv_axis.max(), step=0.5))
        ax.tick_params(labelsize=14)

        ax.text(
            0.0,
            1.16,
            PRETTY_ATTR.get(attr, attr),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=ATTRIBUTE_COLORS[attr],
        )

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if quantity == "shift":
            colorbar.set_label("Average utility shift", fontsize=14)
        elif quantity == "post":
            colorbar.set_label("Average post-event utility", fontsize=14)
        else:
            colorbar.set_label("Average pre-event utility", fontsize=14)
        colorbar.ax.tick_params(labelsize=12)

    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    return fig


def plot_h4(idata, output, conjoint_config):
    posterior = idata.posterior
    if "gamma_fv" not in posterior or "shift_fin" not in posterior:
        raise ValueError("H4 requires a main model fitted with financial vulnerability.")
    if not hasattr(idata, "constant_data") or "fv_data" not in idata.constant_data:
        raise ValueError("H4 requires 'fv_data' in idata.constant_data.")

    baselines = _baselines(conjoint_config)
    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_old_level_order(levels_by_attr)

    # This is the final selected H4 panel from the old analyze_results.ipynb.
    levels_h4 = {
        "costs": [
            "People & companies being protected by protective measures",
        ],
    }

    nhv_to_model_sign = 1
    fv_to_model_sign = 1
    eta_nhv_mean = posterior["eta_nhv"].mean(("chain", "draw")).values
    fv_values = np.asarray(idata.constant_data["fv_data"].values, dtype=float)

    nhv_axis = make_axis_from_values(
        eta_nhv_mean, sign=1, n=80, pad_frac=0.03, use_quantiles=True
    )
    fv_axis = make_axis_from_values(
        fv_values, sign=1, n=80, pad_frac=0.03, use_quantiles=True
    )

    fig = plot_multi_level_heatmaps_flat(
        idata=idata,
        levels_h4=levels_h4,
        levels_by_attr=levels_by_attr,
        baselines=baselines,
        nhv_axis=nhv_axis,
        fv_axis=fv_axis,
        quantity="shift",
        pd_fixed=0.0,
        nhv_to_model_sign=nhv_to_model_sign,
        fv_to_model_sign=fv_to_model_sign,
    )
    save_figure(fig, output)


def main(snakemake) -> None:
    idata = az.from_netcdf(snakemake.input.model)
    conjoint_config = snakemake.params.conjoint

    plotters = {
        "H1": plot_h1,
        "H2": plot_h2,
        "H3": plot_h3,
        "H4": plot_h4,
    }

    for figure in snakemake.params.figures:
        hypothesis = figure["hypothesis"]
        if hypothesis not in plotters:
            raise ValueError(f"Unknown hypothesis plot: {hypothesis}")
        plotters[hypothesis](idata, figure["output"], conjoint_config)
        print(f"Saved {hypothesis}: {figure['output']}")


if __name__ == "__main__":
    main(globals()["snakemake"])
