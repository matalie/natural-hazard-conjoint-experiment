"""Hypothesis-specific plotting logic for H1-H4.

This module contains the scientific/posterior-to-figure logic only.
It does not know about Snakemake paths or output files.
"""
from __future__ import annotations

import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .posterior import (
    build_level_summary_from_da,
    check_model_coding,
    compute_level_surface,
)
from .plotting import (
    ATTRIBUTE_COLORS,
    ATTRIBUTE_ORDER,
    LEVEL_ORDER,
    PRETTY_ATTRIBUTES,
    TEAL_BROWN,
    plot_forest_panels,
)

def _levels_from_config(conjoint_config):
    return {
        attr: list(conjoint_config["attributes"][attr]["levels"])
        for attr in ATTRIBUTE_ORDER
    }

def _validate_level_order(levels_by_attr):
    """Fail loudly if the conjoint design no longer matches the legacy figure order."""
    for attr in ATTRIBUTE_ORDER:
        if set(LEVEL_ORDER[attr]) != set(levels_by_attr[attr]):
            raise ValueError(
                f"Configured levels for '{attr}' do not match the legacy plotting order. "
                "Update LEVEL_ORDER in plotting.py if the conjoint design changed."
            )

def _subplot_color_map(kinds):
    return {
        (attr, kind): ATTRIBUTE_COLORS[attr]
        for attr in ATTRIBUTE_ORDER
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

def plot_h1(idata, conjoint_config,):
    """H1: population pre-event utility and average event-related utility shift."""
    posterior = idata.posterior
    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_level_order(levels_by_attr)

    pre_total_da = posterior["partworth_mean"]
    shift_total_da = posterior["shift_mean"]

    pre_df, pre_draws = build_level_summary_from_da(pre_total_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="pre_total",)
    shift_df, shift_draws = build_level_summary_from_da(shift_total_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="shift_total",)

    plot_df = pd.concat([pre_df, shift_df], ignore_index=True)
    draws_map = {**pre_draws, **shift_draws}
    pre_xlim = _padded_xlim(plot_df, ["pre_total"])

    panels = [
        {
            "kinds": ["pre_total"],
            "title": "Pre-event average utility\n$\\beta_{k}$",
            "xlabel": "Pre-event average utility\n$\\beta_{k}$",
            "xlim": pre_xlim,
            "zero": False,
        },
        {
            "kinds": ["shift_total"],
            "title": "Average utility shift\n$\\delta_{k}$",
            "xlabel": "Average utility shift\n$\\delta_{k}$",
            "xlim": pre_xlim,
            "zero": True,
        },
    ]

    fig, _ = plot_forest_panels(
        plot_df=plot_df,
        attr_order=ATTRIBUTE_ORDER,
        levels_by_attr=LEVEL_ORDER,
        panels=panels,
        subplot_color_map=_subplot_color_map(["pre_total", "shift_total"]),
        draws_map=draws_map,
        show_density=True,
        fig_width=11,
        left_margin=0.45,
        per_level_height=0.40,
        level_wrap_width=50,
    )
    return fig, plot_df


def _build_h2(idata, conjoint_config, pd_sign=1.0, plot_config=None):
    posterior = idata.posterior
    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_level_order(levels_by_attr)

    # Legacy analyze_results.ipynb used pd_sign = +1.
    # Use -1 only if the substantive definition of eta_pd is intentionally reversed.
    gamma_pd_da = float(pd_sign) * posterior["gamma_pd"]
    shift_psy_da = float(pd_sign) * posterior["shift_psy"]
    post_pd_da = gamma_pd_da + shift_psy_da
    
    pd_pre_df, pd_pre_draws = build_level_summary_from_da(gamma_pd_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="pd_pre",)
    pd_shift_df, pd_shift_draws = build_level_summary_from_da(shift_psy_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="pd_shift",)
    pd_post_df, pd_post_draws = build_level_summary_from_da(post_pd_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="pd_post",)

    plot_df = pd.concat([pd_pre_df, pd_shift_df, pd_post_df], ignore_index=True)
    draws_map = {**pd_pre_draws, **pd_shift_draws, **pd_post_draws}
    xlim = _padded_xlim(
        plot_df,
        ["pd_pre", "pd_shift", "pd_post"],
        pad_fraction=0.0,
    )
    return plot_df, draws_map, xlim


def plot_h2(idata, conjoint_config, plot_config=None):
    """H2: psychological-distance associations before the event and with the shift."""
    plot_config = plot_config or {}
    pd_sign = float(plot_config.get("pd_sign", 1.0))

    plot_df, draws_map, h2_xlim = _build_h2(
        idata,
        conjoint_config,
        pd_sign=pd_sign,
    )

    panels = [
        {
            "kinds": ["pd_pre"],
            "title": "Pre-event utility association\n$\\gamma_{PD}$",
            "xlabel": "Pre-event utility association\n$\\gamma_{PD}$",
            "xlim": h2_xlim,
            "zero": True,
        },
        {
            "kinds": ["pd_shift"],
            "title": "Utility shift association\n$\\phi_{PD}$",
            "xlabel": "Utility shift association\n$\\phi_{PD}$",
            "xlim": h2_xlim,
            "zero": True,
        },
    ]

    fig, _ = plot_forest_panels(
        plot_df=plot_df,
        attr_order=ATTRIBUTE_ORDER,
        levels_by_attr=LEVEL_ORDER,
        panels=panels,
        subplot_color_map=_subplot_color_map(["pd_pre", "pd_shift", "pd_post"]),
        draws_map=draws_map,
        show_density=True,
        fig_width=11,
        left_margin=0.45,
        per_level_height=0.40,
        level_wrap_width=50,
    )
    return fig, plot_df


def plot_h3(idata, conjoint_config, plot_config=None):
    """H3: financial-vulnerability association with the event-related utility shift."""
    posterior = idata.posterior
    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_level_order(levels_by_attr)

    if "gamma_fv" not in posterior or "shift_fin" not in posterior:
        raise ValueError("H3 requires a main model fitted with financial vulnerability.")

    gamma_fv_da = posterior["gamma_fv"]
    shift_fin_da = posterior["shift_fin"]
    post_fv_da = gamma_fv_da + shift_fin_da
    
    fin_pre_df, fin_pre_draws = build_level_summary_from_da(gamma_fv_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="fin_pre",)
    fin_shift_df, fin_shift_draws = build_level_summary_from_da(shift_fin_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="fin_shift",)
    fin_post_df, fin_post_draws = build_level_summary_from_da(post_fv_da, ATTRIBUTE_ORDER, levels_by_attr, conjoint_config, kind_label="fin_post",)

    plot_df = pd.concat(
        [fin_pre_df, fin_shift_df, fin_post_df],
        ignore_index=True,
    )
    draws_map = {**fin_pre_draws, **fin_shift_draws, **fin_post_draws}

    # The legacy notebook deliberately reused H2's x range for H3.
    _, _, h2_xlim = _build_h2(idata, conjoint_config, pd_sign=1.0)

    panels = [
        {
            "kinds": ["fin_pre"],
            "title": "Pre-event utility assiciation\n$\\gamma_{FV}$",
            "xlabel": "Pre-event utility association\n$\\gamma_{FV}$",
            "xlim": h2_xlim,
            "zero": True,
        },
        {
            "kinds": ["fin_shift"],
            "title": "Utility shift association\n$\\phi_{FV}$",
            "xlabel": "Utility shift association\n$\\phi_{FV}$",
            "xlim": h2_xlim,
            "zero": True,
        },
    ]

    fig, _ = plot_forest_panels(
        plot_df=plot_df,
        attr_order=ATTRIBUTE_ORDER,
        levels_by_attr=LEVEL_ORDER,
        panels=panels,
        subplot_color_map=_subplot_color_map(["fin_pre", "fin_shift", "fin_post"]),
        draws_map=draws_map,
        show_density=True,
        fig_width=11,
        left_margin=0.45,
        per_level_height=0.40,
        level_wrap_width=50,
    )
    return fig, plot_df


def make_axis_from_values(values, sign=1, n=80, pad_frac=0.03, use_quantiles=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot create an axis from empty/non-finite values.")

    axis_values = float(sign) * values
    if use_quantiles:
        lo, hi = np.quantile(axis_values, [0.055, 0.945])
    else:
        lo, hi = np.min(axis_values), np.max(axis_values)

    if hi == lo:
        return np.linspace(lo - 0.1, hi + 0.1, n)

    pad = pad_frac * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n)


def make_regular_ticks(lo, hi, step=0.5):
    start = np.ceil(lo / step) * step
    stop = np.floor(hi / step) * step
    ticks = np.arange(start, stop + step * 0.5, step)
    return np.round(ticks, 10)


def _wrap_title(text, width=38):
    return "\n".join(
        textwrap.wrap(str(text), width=width, break_long_words=False)
    )


def _resolve_h4_levels(plot_config, levels_by_attr):
    """Resolve the H4 level subset while preserving config order."""
    requested = (plot_config or {}).get("levels", "all")

    if requested in (None, "all"):
        return {
            attr: [level for level in LEVEL_ORDER[attr] if level in levels_by_attr[attr]]
            for attr in ATTRIBUTE_ORDER
        }

    if not isinstance(requested, dict):
        raise ValueError("H4 'levels' must be 'all' or a mapping of attribute -> levels.")

    resolved = {}
    for attr, requested_levels in requested.items():
        if attr not in levels_by_attr:
            raise ValueError(f"Unknown H4 attribute: {attr}")

        # Empty YAML entry, e.g. `costs:`, means: do not plot this attribute.
        if requested_levels is None:
            continue

        # Optional convenience: `costs: all`
        if requested_levels == "all":
            requested_levels = list(levels_by_attr[attr])
        elif isinstance(requested_levels, str):
            requested_levels = [requested_levels]
        else:
            requested_levels = list(requested_levels)

        unknown = [level for level in requested_levels if level not in levels_by_attr[attr]]
        if unknown:
            raise ValueError(
                f"Unknown H4 levels for '{attr}': {unknown}. "
                f"Valid levels are: {levels_by_attr[attr]}"
            )

        if requested_levels:
            resolved[attr] = requested_levels

    if not resolved:
        raise ValueError("H4 level selection is empty.")

    return resolved

def _plot_multi_level_heatmaps(
    idata, levels_h4, conjoint_config, nhv_axis, fv_axis,
    quantity="shift", pd_fixed=0.0, panel_width=3.7, panel_height=3.8,
):
    """Plot selected H4 levels with one row per attribute."""
    attrs = [a for a in ATTRIBUTE_ORDER if a in levels_h4 and levels_h4[a]]
    if not attrs:
        raise ValueError("H4 needs at least one heatmap panel.")

    surfaces, table_rows, scale_values = {}, [], []

    for attr in attrs:
        for level in levels_h4[attr]:
            z_mean, z_low, z_high = compute_level_surface(
                idata=idata, attr=attr, level=level,
                conjoint_config=conjoint_config,
                nhv_axis=nhv_axis, fv_axis=fv_axis,
                quantity=quantity, pd_fixed=pd_fixed,
            )
            surfaces[(attr, level)] = z_mean
            scale_values += [float(np.nanmin(z_mean)), float(np.nanmax(z_mean))]

            for i, fv in enumerate(fv_axis):
                for j, nhv in enumerate(nhv_axis):
                    table_rows.append({
                        "attribute": attr,
                        "level": level,
                        "nhv": float(nhv),
                        "fv": float(fv),
                        "mean": float(z_mean[i, j]),
                        "hdi_low": float(z_low[i, j]),
                        "hdi_high": float(z_high[i, j]),
                        "quantity": quantity,
                    })

    low, high = min(scale_values), max(scale_values)
    if low < 0 < high:
        vmax = max(abs(low), abs(high))
        vmin = -vmax
    else:
        vmin, vmax = low, high
        if vmin == vmax:
            vmin -= 0.01
            vmax += 0.01

    max_levels = max(len(levels_h4[a]) for a in attrs)
    nrows = len(attrs)
    fig = plt.figure(figsize=(panel_width * max_levels + 1.6, panel_height * nrows + 0.6))
    outer = fig.add_gridspec( nrows, 1, left=0.08, right=0.86, top=0.96, bottom=0.07, hspace=0.55,)

    image = None
    row_axes_map = {}
    quantity_label = {
        "shift": "Average utility shift",
        "post": "Average post-event utility",
        "pre": "Average pre-event utility",
    }.get(quantity, quantity)

    for row, attr in enumerate(attrs):
        levels = levels_h4[attr]
        inner = outer[row, 0].subgridspec(
            3, len(levels),
            height_ratios=[0.13, 0.24, 1.0],
            hspace=0.03,
            wspace=0.28,
        )

        # 1) Gruppentitel über der gesamten Attributgruppe
        group_ax = fig.add_subplot(inner[0, :])
        group_ax.axis("off")
        group_ax.text(
            0.5, 0.5, PRETTY_ATTRIBUTES.get(attr, attr),
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="black",
        )

        row_axes = []

        for col, level in enumerate(levels):
            # 2) Leveltitel in eigener Achse
            title_ax = fig.add_subplot(inner[1, col])
            title_ax.axis("off")
            title_ax.text(
                0.5, 0.5, _wrap_title(level, width=24),
                ha="center", va="center",
                fontsize=12, color="black",
            )

            # 3) Heatmap darunter
            ax = fig.add_subplot(inner[2, col])
            image = ax.imshow(
                surfaces[(attr, level)],
                origin="lower",
                aspect="auto",
                extent=[nhv_axis.min(), nhv_axis.max(), fv_axis.min(), fv_axis.max()],
                cmap=TEAL_BROWN,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_box_aspect(0.92)
            ax.set_xlabel("Natural hazard vulnerability", fontsize=11)

            if col == 0:
                ax.set_ylabel("Financial vulnerability", fontsize=11)
            else:
                ax.tick_params(axis="y", labelleft=False)

            ax.set_xticks(make_regular_ticks(nhv_axis.min(), nhv_axis.max(), step=0.5))
            ax.set_yticks(make_regular_ticks(fv_axis.min(), fv_axis.max(), step=0.5))
            ax.tick_params(labelsize=9.5)
            row_axes.append(ax)

        row_axes_map[attr] = row_axes
        
    if image is not None:
        middle_attr = attrs[len(attrs) // 2]
        ref_ax = row_axes_map[middle_attr][0]
        ref_box = ref_ax.get_position()

        cax = fig.add_axes([0.89, ref_box.y0, 0.025, ref_box.height,])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(quantity_label, fontsize=11)
        cbar.ax.tick_params(labelsize=9.5)

    return fig, pd.DataFrame(table_rows)

def plot_h4(idata, conjoint_config, plot_config=None):
    """H4: additive NHV/FV utility surface for any requested conjoint levels."""
    plot_config = plot_config or {}
    posterior = idata.posterior

    if "gamma_fv" not in posterior or "shift_fin" not in posterior:
        raise ValueError("H4 requires a main model fitted with financial vulnerability.")
    if not hasattr(idata, "constant_data") or "fv_data" not in idata.constant_data:
        raise ValueError("H4 requires 'fv_data' in idata.constant_data.")

    levels_by_attr = _levels_from_config(conjoint_config)
    _validate_level_order(levels_by_attr)
    levels_h4 = _resolve_h4_levels(plot_config, levels_by_attr)

    eta_nhv_mean = posterior["eta_nhv"].mean(("chain", "draw")).values
    fv_values = np.asarray(idata.constant_data["fv_data"].values, dtype=float)

    axis_points = int(plot_config.get("axis_points", 80))
    use_quantiles = bool(plot_config.get("use_quantiles", True))
    nhv_axis = make_axis_from_values(
        eta_nhv_mean,
        sign=1,
        n=axis_points,
        pad_frac=0.03,
        use_quantiles=use_quantiles,
    )
    fv_axis = make_axis_from_values(
        fv_values,
        sign=1,
        n=axis_points,
        pad_frac=0.03,
        use_quantiles=use_quantiles,
    )

    return _plot_multi_level_heatmaps(
        idata=idata,
        levels_h4=levels_h4,
        conjoint_config=conjoint_config,
        nhv_axis=nhv_axis,
        fv_axis=fv_axis,
        quantity=plot_config.get("quantity", "shift"),
        pd_fixed=float(plot_config.get("pd_fixed", 0.0)),
        panel_width=float(plot_config.get("panel_width", 3.0)),
        panel_height=float(plot_config.get("panel_height", 2.8)),
    )


def plot_hypothesis(idata, hypothesis, conjoint_config, plot_config=None):
    """Dispatch one configured hypothesis plot and return its Matplotlib figure."""
    check_model_coding(idata, conjoint_config)
    plotters = {
        "H1": plot_h1,
        "H2": plot_h2,
        "H3": plot_h3,
        "H4": plot_h4,
    }
    try:
        plotter = plotters[hypothesis]
    except KeyError as exc:
        raise ValueError(f"Unknown hypothesis plot: {hypothesis}") from exc

    return plotter(idata, conjoint_config, plot_config=plot_config)
