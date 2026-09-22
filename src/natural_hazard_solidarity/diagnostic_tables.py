"""Posterior summaries and MCMC diagnostics for the main model."""

from __future__ import annotations

import re

import arviz as az
import pandas as pd


SUMMARY_COLUMNS = ["mean", "sd", "ess_bulk", "ess_tail", "r_hat",]

def _summary(idata, var_names):
    """Return compact posterior summary and sampling diagnostics."""
    table = az.summary(idata, var_names=var_names, kind="all", round_to=None,)
    table = table[SUMMARY_COLUMNS].copy()
    table = table.rename(
        columns={
            "mean": "Mean",
            "sd": "SD",
            "ess_bulk": "ESS bulk",
            "ess_tail": "ESS tail",
            "r_hat": "Rhat",
        }
    )
    return table


def main_model_summary(idata):
    """Summary of population-level parameters in the main HCM."""
    variables = [
        "partworth_mean",
        "gamma_nhv",
        "gamma_pd",
        "gamma_fv",
        "shift_mean",
        "shift_nh",
        "shift_psy",
        "shift_fin",
        "partworth_sd",
        "alpha",
        "likert_sigma",
    ]

    # Only use variables actually present in the model.
    available = set(idata.posterior.data_vars)
    variables = [v for v in variables if v in available]
    table = _summary(idata, variables)

    # Paper-style names
    replacements = {
        "partworth_mean": "β_pre",
        "gamma_nhv": "γ_NHV",
        "gamma_pd": "γ_PD",
        "gamma_fv": "γ_FV",
        "shift_mean": "δ",
        "shift_nh": "δ_NHV",
        "shift_psy": "δ_PD",
        "shift_fin": "δ_FV",
        "partworth_sd": "σ_β",
        "likert_sigma": "σ_Likert",
        "alpha": "α",
    }

    labels = []
    for parameter in table.index.astype(str):
        label = parameter
        for old, new in replacements.items():
            if label == old or label.startswith(old + "["):
                label = new + label[len(old):]
                break
        labels.append(label)

    table.index = labels
    table.index.name = "Parameter"

    return table.reset_index()


def latent_loading_summary(idata):
    """Sampling diagnostics for NHV and psychological-distance loadings."""

    table = _summary(idata, ["loading_nhv", "loading_pd",],)
    labels = []

    for parameter in table.index.astype(str):
        # e.g. loading_nhv[S0_sensitivity_nh_1] -> λ_NHV[NHV1]
        match = re.match(r"loading_nhv\[.*_(\d+)\]$", parameter)
        if match:
            labels.append(f"λ_NHV[NHV{match.group(1)}]")
            continue

        # e.g. loading_pd[S1_psycho_distance_4] -> λ_PD[PD4]
        match = re.match(r"loading_pd\[.*_(\d+)\]$", parameter)
        if match:
            labels.append(f"λ_PD[PD{match.group(1)}]")
            continue
        labels.append(parameter)

    table.index = labels
    table.index.name = "Parameter"

    return table.reset_index()