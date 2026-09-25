"""Exploratory factor-analysis and construct-correlation figures"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .plotting import (BROWN_PURPLE)

EFA_ITEMS = [
    "S1_psycho_distance_1",
    "S1_psycho_distance_2",
    "S1_psycho_distance_3",
    "S1_psycho_distance_4",
    "S0_finan_vulnerability_1",
    "S0_costs_cc_policy_1",
    "S0_sensitivity_nh_1",
    "S0_sensitivity_nh_2",
    "S0_sensitivity_nh_3",
]

def complete_efa_items(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in EFA_ITEMS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing EFA columns: {missing}")
    values = df[EFA_ITEMS].apply(pd.to_numeric, errors="coerce")
    return values.dropna().copy()


def plot_correlation_matrix(attributes: pd.DataFrame):
    """Lower-triangle Pearson correlation matrix."""
    correlation_matrix = attributes.corr()
    values = correlation_matrix.values.copy()
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    masked = np.ma.array(values, mask=mask)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, vmin=-1, vmax=1, cmap=BROWN_PURPLE)
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if not mask[i, j]:
                ax.text(j, i, f"{correlation_matrix.iat[i, j]:.2f}", ha="center", va="center", fontsize=10)

    labels = [re.sub(r"^S\d+_", "", str(value)) for value in correlation_matrix.columns]
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation")
    ax.set_xticks(np.arange(correlation_matrix.shape[1]))
    ax.set_yticks(np.arange(correlation_matrix.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.tight_layout()
    return fig, correlation_matrix


def fit_efa(attributes: pd.DataFrame):
    """Fit the ML/oblimin 2- and 3-factor EFAs."""
    try:
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    except ImportError as exc:
        raise ImportError("EFA plots require the factor-analyzer package; use workflow/envs/environment.yaml.") from exc

    raw = attributes.to_numpy(dtype=float)
    means = raw.mean(axis=0)
    sds = raw.std(axis=0, ddof=0)
    if np.any(~np.isfinite(sds)) or np.any(sds == 0):
        raise ValueError("EFA items must have finite non-zero variance.")
    zscores = pd.DataFrame((raw - means) / sds, columns=attributes.columns, index=attributes.index)

    _, kmo_model = calculate_kmo(zscores)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(zscores)
    eigvals = np.sort(np.linalg.eigvals(zscores.corr().values).real)[::-1]

    results = {}
    for n_factors in (2, 3):
        fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="ml")
        fa.fit(zscores)
        factor_names = [f"F{i}" for i in range(1, n_factors + 1)]
        loadings = pd.DataFrame(fa.loadings_, index=attributes.columns, columns=factor_names)
        phi = getattr(fa, "phi_", None)
        if phi is None:
            phi = np.eye(n_factors)
        phi_df = pd.DataFrame(phi, index=factor_names, columns=factor_names)
        results[n_factors] = {"loadings": loadings, "phi_df": phi_df}

    diagnostics = {
        "n_complete": len(zscores),
        "kmo_model": float(kmo_model),
        "bartlett_chi2": float(bartlett_chi2),
        "bartlett_p": float(bartlett_p),
        "eigenvalues": eigvals,
    }
    return results, diagnostics


def plot_efa_comparison(efa_results: dict):
    """2-vs-3 factor figure with physically square heatmap cells."""
    cell = 0.5
    n_items = len(efa_results[2]["loadings"])
    n_fac_2, n_fac_3 = 2, 3

    phi_w2 = phi_h2 = n_fac_2 * cell
    phi_w3 = phi_h3 = n_fac_3 * cell
    load_w2, load_w3, load_h = n_fac_2 * cell, n_fac_3 * cell, n_items * cell
    left_margin, right_margin, bottom_margin, top_margin = 2.8, 0.7, 0.8, 0.8
    x_gap, y_gap, cbar_gap, cbar_w = 0.9, 0.6, 0.15, 0.18
    x2 = left_margin
    x3 = x2 + load_w2 + x_gap
    cbar_x = x3 + load_w3 + cbar_gap
    y_load = bottom_margin
    y_phi = y_load + load_h + y_gap
    fig_w = left_margin + load_w2 + x_gap + load_w3 + cbar_gap + cbar_w + right_margin
    fig_h = bottom_margin + load_h + y_gap + phi_h3 + top_margin
    fs = 12
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig = plt.figure(figsize=(fig_w, fig_h))

    def add_axes_inch(x, y, width, height, **kwargs):
        return fig.add_axes([x / fig_w, y / fig_h, width / fig_w, height / fig_h], **kwargs)

    ax_phi_2 = add_axes_inch(x2, y_phi, phi_w2, phi_h2)
    ax_phi_3 = add_axes_inch(x3, y_phi, phi_w3, phi_h3)
    ax_load_2 = add_axes_inch(x2, y_load, load_w2, load_h)
    ax_load_3 = add_axes_inch(x3, y_load, load_w3, load_h, sharey=ax_load_2)
    cax_phi = add_axes_inch(cbar_x, y_phi, cbar_w, phi_h3)
    cax_load = add_axes_inch(cbar_x, y_load, cbar_w, load_h)

    def plot_phi(ax, phi_df, title):
        mask = np.triu(np.ones_like(phi_df.values, dtype=bool), k=1)
        masked = np.ma.array(phi_df.values, mask=mask)
        im = ax.imshow(masked, aspect="equal", cmap=BROWN_PURPLE, norm=norm, interpolation="none")
        ax.set_title(title, pad=8, fontsize=fs)
        ax.set_xticks(range(phi_df.shape[1]))
        ax.set_xticklabels(phi_df.columns, ha="right", fontsize=fs)
        ax.set_yticks(range(phi_df.shape[0]))
        ax.set_yticklabels(phi_df.index, fontsize=fs)
        for i in range(phi_df.shape[0]):
            for j in range(phi_df.shape[1]):
                if not mask[i, j]:
                    ax.text(j, i, f"{phi_df.iat[i, j]:.2f}", ha="center", va="center", fontsize=fs)
        return im

    def plot_loadings(ax, loadings, title, show_ylabels=True):
        im = ax.imshow(loadings.values, aspect="equal", cmap=BROWN_PURPLE, norm=norm, interpolation="none")
        ylabels = [re.sub(r"^S\d+_", "", str(value)) for value in loadings.index]
        ax.set_title(title, pad=8, fontsize=fs)
        ax.set_xticks(range(loadings.shape[1]))
        ax.set_xticklabels(loadings.columns, ha="right", fontsize=fs)
        ax.set_yticks(range(loadings.shape[0]))
        ax.set_yticklabels(ylabels, fontsize=fs)
        if not show_ylabels:
            ax.tick_params(labelleft=False)
        for i in range(loadings.shape[0]):
            for j in range(loadings.shape[1]):
                ax.text(j, i, f"{loadings.iat[i, j]:.2f}", ha="center", va="center", fontsize=fs)
        return im

    plot_phi(ax_phi_2, efa_results[2]["phi_df"], "Φ — 2 factors")
    im_phi_3 = plot_phi(ax_phi_3, efa_results[3]["phi_df"], "Φ — 3 factors")
    plot_loadings(ax_load_2, efa_results[2]["loadings"], "2 factor loadings", show_ylabels=True)
    im_load_3 = plot_loadings(ax_load_3, efa_results[3]["loadings"], "3 factor loadings", show_ylabels=False)

    cbar_phi = fig.colorbar(im_phi_3, cax=cax_phi)
    cbar_phi.set_label("Factor correlation Φ", fontsize=fs)
    cbar_phi.ax.tick_params(labelsize=10)
    cbar_loading = fig.colorbar(im_load_3, cax=cax_load)
    cbar_loading.set_label("Factor loading", fontsize=fs)
    cbar_loading.ax.tick_params(labelsize=10)
    return fig
