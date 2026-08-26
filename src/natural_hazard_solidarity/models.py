"""Preparation and PyMC definitions for the final HCM and its adjusted variant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from .data_preparation import design_columns


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=0, keepdims=True)
    sd = np.nanstd(values, axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (values - mean) / sd


def prepare_model_data(df: pd.DataFrame, conjoint_config: dict, constructs: dict) -> dict:
    """Convert encoded conjoint data into arrays for the PyMC model."""

    features = design_columns(conjoint_config)
    ordered = df.sort_values(["task_id", "option"]).copy()

    if not ordered.groupby("task_id").size().eq(2).all():
        raise ValueError("Each task must contain two alternatives.")
    if not ordered.groupby("task_id")["chosen"].sum().eq(1).all():
        raise ValueError("Each task must contain one chosen alternative.")

    left_option = conjoint_config.get("left_option", 1)
    right_option = conjoint_config.get("right_option", 2)

    left = ordered["option"].eq(left_option)
    right = ordered["option"].eq(right_option)

    if not np.array_equal(ordered.loc[left, "task_id"].to_numpy(), ordered.loc[right, "task_id"].to_numpy(),):
        raise ValueError("Left and right alternatives are not aligned.")

    x_left = ordered.loc[left, features].to_numpy(float)
    x_right = ordered.loc[right, features].to_numpy(float)
    y_left = ordered.loc[left, "chosen"].to_numpy(int)
    event = ordered.loc[left, "nh_event"].to_numpy(int)

    respondent_index, respondent_ids = pd.factorize(ordered.loc[left, "respondent_id"], sort=False)

    nhv_columns = constructs["natural_hazard_vulnerability"]
    pd_columns = constructs["psychological_distance"]
    fv_column = constructs["financial_vulnerability"]

    respondent_level = (
        ordered[["respondent_id", *nhv_columns, *pd_columns, fv_column]]
        .drop_duplicates("respondent_id")
        .set_index("respondent_id")
        .reindex(respondent_ids)
    )

    y_nhv = respondent_level[nhv_columns].to_numpy(float)
    y_pd = respondent_level[pd_columns].to_numpy(float)
    fv = respondent_level[fv_column].to_numpy(float)

    fv_sd = fv.std(ddof=0)

    return {
        "feature_names": features,
        "nhv_item_names": nhv_columns,
        "pd_item_names": pd_columns,
        "n_respondents": len(respondent_ids),
        "x_left": x_left,
        "x_right": x_right,
        "y_left": y_left,
        "event": event,
        "respondent_index": respondent_index,
        "y_nhv_z": _zscore_columns(y_nhv),
        "y_pd_z": _zscore_columns(y_pd),
        "fv_z": (fv - fv.mean()) / (fv_sd if fv_sd else 1.0),
    }


def build_hcm(data: dict, prior_factor: float, include_financial_vulnerability: bool) -> pm.Model:
    """Final 'easy factor' HCM consolidated from the duplicated complete-HCM notebooks."""
    coords = {
        "task": np.arange(data["x_left"].shape[0]),
        "level": data["feature_names"],
        "respondent": np.arange(data["n_respondents"]),
        "nhv_item": data["nhv_item_names"],
        "pd_item": data["pd_item_names"],
    }
    factor = float(prior_factor)

    with pm.Model(coords=coords) as model:
        x_left = pm.Data("x_left", data["x_left"], dims=("task", "level"))
        x_right = pm.Data("x_right", data["x_right"], dims=("task", "level"))
        choice_left = pm.Data("choice_left", data["y_left"], dims="task")
        event = pm.Data("event", data["event"], dims="task")
        respondent_index = pm.Data("respondent_index", data["respondent_index"], dims="task")

        nhv_data = pm.Data("nhv_data", data["y_nhv_z"], dims=("respondent", "nhv_item"))
        pd_data = pm.Data("pd_data", data["y_pd_z"], dims=("respondent", "pd_item"))
        if include_financial_vulnerability:
            fv_data = pm.Data("fv_data", data["fv_z"], dims="respondent")

        eta_nhv = pm.Normal("eta_nhv", 0, 0.3 * factor, dims="respondent")
        eta_pd = pm.Normal("eta_pd", 0, 0.3 * factor, dims="respondent")
        loading_nhv = pm.HalfNormal("loading_nhv", 0.5 * factor, dims="nhv_item")
        loading_pd = pm.HalfNormal("loading_pd", 0.5 * factor, dims="pd_item")
        likert_sigma = pm.HalfNormal("likert_sigma", 0.5)

        for index in range(len(coords["nhv_item"])):
            pm.Normal(
                f"nhv_like_{index}",
                mu=loading_nhv[index] * eta_nhv,
                sigma=likert_sigma,
                observed=nhv_data[:, index],
                dims="respondent",
            )
        for index in range(len(coords["pd_item"])):
            pm.Normal(
                f"pd_like_{index}",
                mu=loading_pd[index] * eta_pd,
                sigma=likert_sigma,
                observed=pd_data[:, index],
                dims="respondent",
            )

        alpha = pm.Normal("alpha", mu=0.0, sigma=0.3 * factor)
        partworth_mean = pm.Normal("partworth_mean", mu=0.0, sigma=0.3 * factor, dims="level")
        partworth_sd = pm.HalfNormal("partworth_sd", 0.5 * factor, dims="level")
        partworth_z = pm.Normal(
            "partworth_z",
            mu=0.0,
            sigma=0.3 * factor,
            dims=("respondent", "level"),
        )
        gamma_nhv = pm.Normal("gamma_nhv", mu=0.0, sigma=0.2 * factor, dims="level")
        gamma_pd = pm.Normal("gamma_pd", mu=0.0, sigma=0.2 * factor, dims="level")

        pre = partworth_mean + partworth_z * partworth_sd + gamma_nhv * eta_nhv[:, None] + gamma_pd * eta_pd[:, None]
        if include_financial_vulnerability:
            gamma_fv = pm.Normal("gamma_fv", mu=0.0, sigma=0.2 * factor, dims="level")
            pre = pre + gamma_fv * fv_data[:, None]

        partworth_individual = pm.Deterministic(
            "partworth_individual",
            pre,
            dims=("respondent", "level"),
        )

        shift_mean = pm.Normal("shift_mean", mu=0.0, sigma=0.3 * factor, dims="level")
        shift_nh = pm.Normal("shift_nh", mu=0.0, sigma=0.2 * factor, dims="level")
        shift_psy = pm.Normal("shift_psy", mu=0.0, sigma=0.2 * factor, dims="level")
        shift = shift_mean + shift_nh * eta_nhv[:, None] + shift_psy * eta_pd[:, None]
        if include_financial_vulnerability:
            shift_fin = pm.Normal("shift_fin", mu=0.0, sigma=0.2 * factor, dims="level")
            shift = shift + shift_fin * fv_data[:, None]

        shift_individual = pm.Deterministic(
            "shift_eff",
            shift,
            dims=("respondent", "level"),
        )
        task_partworth = (
            partworth_individual[respondent_index]
            + event[:, None] * shift_individual[respondent_index]
        )
        utility_left = pm.math.sum(x_left * task_partworth, axis=1)
        utility_right = pm.math.sum(x_right * task_partworth, axis=1)
        logit_p_left = pm.Deterministic(
            "logit_p_left",
            alpha + utility_left - utility_right,
            dims="task",
        )
        pm.Bernoulli("choice", logit_p=logit_p_left, observed=choice_left, dims="task")

        pm.Deterministic("partworth_pre_mean", partworth_mean, dims="level")
        pm.Deterministic("partworth_post_mean", partworth_mean + shift_mean, dims="level")

    return model
