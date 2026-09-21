"""Respondent-level figures migrated from ``analyze_respondents.ipynb``.

This module contains plotting logic only. File I/O belongs to the Snakemake
wrapper in ``workflow/scripts/plot_respondents.py``.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .plotting import (VERY_LIGHT_PURPLE, LIGHT_PURPLE, PURPLE, LIGHT_TEAL, TEAL)

def _financial_values(idata):
    if not hasattr(idata, "constant_data") or "fv_data" not in idata.constant_data:
        raise ValueError("Respondent plots require fv_data stored in the fitted main model.")
    posterior_ids = idata.posterior.coords["respondent"].values
    fv = idata.constant_data["fv_data"]
    if "respondent" in fv.coords:
        fv = fv.sel(respondent=posterior_ids)
    values = np.asarray(fv.values, dtype=float).reshape(-1)
    if len(values) != len(posterior_ids):
        raise ValueError("fv_data and posterior respondent dimensions do not match.")
    return values


def plot_latent_constructs(idata):
    """Return the legacy eta_PD / eta_NHV / FV histogram figure."""
    eta_pd = idata.posterior["eta_pd"].mean(("chain", "draw")).values
    eta_nhv = idata.posterior["eta_nhv"].mean(("chain", "draw")).values
    fv = _financial_values(idata)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(eta_pd, color=VERY_LIGHT_PURPLE, bins=20, edgecolor="black")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title(r"Psychological distance: $\eta_{PD}$", fontsize=14)
    axes[0].set_xlabel(r"$\eta_{PD}$", fontsize=14)

    axes[1].hist(eta_nhv, color=LIGHT_PURPLE, bins=20, edgecolor="black")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title(r"Perceived natural hazard vulnerability: $\eta_{NHV}$", fontsize=14)
    axes[1].set_xlabel(r"$\eta_{NHV}$", fontsize=14)

    axes[2].hist(fv, color=PURPLE, bins=20, edgecolor="black")
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title(r"Perceived financial vulnerability: $FV$", fontsize=14)
    axes[2].set_xlabel(r"$FV$", fontsize=14)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)
        ax.set_ylabel("Respondents", fontsize=14)
    fig.tight_layout()
    return fig


def plot_level_decomposition_by_respondent(decomp_df, title=None):
    """Return the legacy four-panel respondent-impact figure.

    Each panel is sorted independently, as in the original notebook.
    """
    subplot_titles = ["Perceived NHV", "Psychological distance", "Perceived FV", "Total shift"]
    ylabels = [
        r"$\delta_{eff,n}$ contribution",
        r"$\delta_{eff,n}$ contribution",
        r"$\delta_{eff,n}$ contribution",
        r"Total $\delta_{eff,n}$",
    ]
    xlabels = [
        "Respondents sorted by\nNHV contribution",
        "Respondents sorted by\nPD contribution",
        "Respondents sorted by\nFV contribution",
        "Respondents sorted by\ntotal shift",
    ]
    panels = [
        ("nhv_contrib_mean", "nhv_contrib_hdi_low", "nhv_contrib_hdi_high", "nhv_contrib_mean"),
        ("pd_contrib_mean", "pd_contrib_hdi_low", "pd_contrib_hdi_high", "pd_contrib_mean"),
        ("fv_contrib_mean", "fv_contrib_hdi_low", "fv_contrib_hdi_high", "fv_contrib_mean"),
        ("total_shift_mean", "total_shift_hdi_low", "total_shift_hdi_high", "total_shift_mean"),
    ]
    fixed_xticks = [0, 200, 400]
    fixed_yticks = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
    fontsize = 17

    fig = plt.figure(figsize=(14, 3.5))
    gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[1, 1, 1, 0.12, 1], wspace=0.08)
    axes = np.array([
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 4]),
    ])

    for i, (ax, (mean_col, low_col, high_col, sort_col)) in enumerate(zip(axes, panels)):
        frame = decomp_df.sort_values(sort_col, ascending=True).reset_index(drop=True)
        x = np.arange(len(frame))
        ax.vlines(
            x,
            frame[low_col],
            frame[high_col],
            color=LIGHT_TEAL,
            alpha=0.3,
            linewidth=0.9,
            zorder=1,
        )
        ax.scatter(x, frame[mean_col], s=6, alpha=0.8, zorder=2, color=TEAL)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_title(subplot_titles[i], fontsize=fontsize)
        if i in (0, 3):
            ax.set_ylabel(ylabels[i], fontsize=fontsize)
        ax.set_yticks(fixed_yticks)
        if i == 0:
            ax.set_yticklabels(fixed_yticks, fontsize=fontsize)
            ax.tick_params(axis="y", labelleft=True, left=True, labelsize=fontsize)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", labelleft=False, left=True)
        valid_xticks = [tick for tick in fixed_xticks if tick <= len(frame) - 1]
        ax.set_xticks(valid_xticks)
        ax.set_xticklabels(valid_xticks, fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.set_xlabel(xlabels[i], fontsize=fontsize)
        ax.set_ylim(-0.8, 0.8)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.80, wspace=0.10)
    return fig
