"""Legacy HCM and two legacy choice baselines; no adjusted model.

Model equations/prior scales are preserved. Coordinates and metadata are not
statistical parameters. PyMC is imported only by builders, not by data preparation.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import hashlib
import json
import numpy as np
import pandas as pd
from .data_preparation import design_columns, validate_choice_tasks
if TYPE_CHECKING:
    import pymc as pm


def _standardize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite construct values before standardization.")
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=0)
    scale = np.where(sd == 0, 1.0, sd)
    return (values - mean) / scale, {"mean": np.asarray(mean).tolist(), "sd": np.asarray(scale).tolist()}


def prepare_model_data(df, conjoint_config, constructs, *, include_constructs=True):
    """Validated paired design, true IDs, and within-run legacy z-standardization."""
    left_code, right_code = conjoint_config.get("left_option", 1), conjoint_config.get("right_option", 2)
    validate_choice_tasks(df, left_code, right_code)
    features = design_columns(conjoint_config)
    ordered = df.sort_values(["task_id", "option"]).copy()
    left = ordered.loc[ordered["option"].eq(left_code)].copy()
    right = ordered.loc[ordered["option"].eq(right_code)].copy()
    if not np.array_equal(left["task_id"], right["task_id"]):
        raise ValueError("Left/right tasks are not aligned.")
    ridx, rids = pd.factorize(left["respondent_id"], sort=False)
    result = {
        "feature_names": features, "task_ids": left["task_id"].astype(str).to_numpy(),
        "respondent_ids": rids.to_numpy(), "n_respondents": len(rids),
        "respondent_index": ridx.astype("int64"),
        "x_left": left[features].to_numpy(float), "x_right": right[features].to_numpy(float),
        "y_left": left["chosen"].to_numpy("int64"), "event": left["nh_event"].to_numpy("int64"),
    }
    if not np.isfinite(result["x_left"]).all() or not np.isfinite(result["x_right"]).all():
        raise ValueError("Non-finite encoded features.")
    # Persistent linking fingerprint avoids unsafe demographic joins after refiltering.
    if "respondent_key" in ordered:
        if not ordered.groupby("respondent_id")["respondent_key"].nunique().eq(1).all():
            raise ValueError("Inconsistent respondent linking keys.")
        result["respondent_keys"] = (ordered.drop_duplicates("respondent_id").set_index("respondent_id")
                                     .loc[rids, "respondent_key"].astype(str).to_numpy())
    else:
        raise KeyError("respondent_key missing: rebuild data with the supplied preprocessing.")

    if include_constructs:
        nh, psy, fv = (constructs[k] for k in ["natural_hazard_vulnerability", "psychological_distance", "financial_vulnerability"])
        columns = list(dict.fromkeys([*nh, *psy, fv]))
        if not ordered.groupby("respondent_id")[columns].nunique(dropna=False).eq(1).all().all():
            raise ValueError("Construct values differ across rows of the same respondent.")
        respondents = ordered.drop_duplicates("respondent_id").set_index("respondent_id").loc[rids, columns]
        result["y_nhv_z"], nh_scale = _standardize(respondents[nh].to_numpy(float))
        result["y_pd_z"], pd_scale = _standardize(respondents[psy].to_numpy(float))
        result["fv_z"], fv_scale = _standardize(respondents[fv].to_numpy(float))
        result["nhv_item_names"], result["pd_item_names"] = nh, psy
        result["standardization"] = {"nhv": {"columns": nh, **nh_scale},
                                     "pd": {"columns": psy, **pd_scale},
                                     "fv": {"column": fv, **fv_scale}}
    else:
        result["standardization"] = {}
    return result


def choice_signature(data):
    """Hash observations/design for strict within-observation model comparisons."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"task_ids": data["task_ids"].tolist(), "features": data["feature_names"],
                              "respondent_keys": data["respondent_keys"].tolist()}, sort_keys=True).encode())
    for name in ["x_left", "x_right", "y_left", "event", "respondent_index"]:
        arr = np.asarray(data[name], dtype="<f8")
        digest.update(str(arr.shape).encode()); digest.update(arr.tobytes())
    return digest.hexdigest()



def build_simple_choice_model(data) -> pm.Model:
    """Test first simple model with parameters alpha (left bias) and a common partworth mean (utility of an option) without event shift."""
    import pymc as pm
    coords = {"level": data["feature_names"], "task": data["task_ids"]}
    with pm.Model(coords=coords) as model:
        left = pm.Data("x_left", data["x_left"], dims=("task", "level"))
        right = pm.Data("x_right", data["x_right"], dims=("task", "level"))
        observed = pm.Data("choice_left", data["y_left"], dims="task")
        alpha = pm.Normal("alpha", 0, 0.3)
        beta = pm.Normal("partworth_mean", 0, 0.5, dims="level")
        logit = pm.Deterministic("logit_p_left", alpha + pm.math.sum((left-right)*beta, axis=1), dims="task")
        pm.Bernoulli("choice", logit_p=logit, observed=observed, dims="task")
    return model


def build_longitudinal_mixed_logit(data) -> pm.Model:
    """Test simple longitudinal choice model with alpha (left bias) and a partworth mean value as well as an individual partworth preference (utility of an option), and a common partworth shift (utility shift)."""
    import pymc as pm
    coords = {"level": data["feature_names"], "task": data["task_ids"], "respondent": data["respondent_ids"]}
    with pm.Model(coords=coords) as model:
        left = pm.Data("x_left", data["x_left"], dims=("task", "level"))
        right = pm.Data("x_right", data["x_right"], dims=("task", "level"))
        observed = pm.Data("choice_left", data["y_left"], dims="task")
        event = pm.Data("event", data["event"], dims="task")
        ridx = pm.Data("respondent_index", data["respondent_index"], dims="task")
        alpha = pm.Normal("alpha", 0, 0.3)
        beta = pm.Normal("partworth_mean", 0, 0.5, dims="level")
        sd = pm.HalfNormal("partworth_sd", 0.5, dims="level")
        z = pm.Normal("partworth_z", 0, 0.3, dims=("respondent", "level"))
        individual = pm.Deterministic("partworth_individual", beta + z*sd, dims=("respondent", "level"))
        shift = pm.Normal("shift", 0, 0.5, dims="level")
        effective = individual[ridx] + shift * event[:, None]
        logit = pm.Deterministic("logit_p_left", alpha + pm.math.sum((left-right)*effective, axis=1), dims="task")
        pm.Bernoulli("choice", logit_p=logit, observed=observed, dims="task")
        pm.Deterministic("partworth_post", beta + shift, dims="level")
    return model


def build_hcm(data: dict, prior_factor: float, include_financial_vulnerability: bool) -> pm.Model:
    """Final 'easy factor' HCM consolidated from the duplicated complete-HCM notebooks."""
    import pymc as pm
    if not np.isfinite(prior_factor) or prior_factor <= 0:
        raise ValueError("prior_factor must be positive and finite.")
    coords = {
        "task": data["task_ids"],
        "level": data["feature_names"],
        "respondent": data["respondent_ids"],
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


