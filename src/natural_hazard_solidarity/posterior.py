"""Posterior transformations and summaries for fitted conjoint models.

Derived quantities are constructed draw by draw before posterior summaries or
intervals are computed. This preserves posterior dependence when coefficients,
respondent effects, or group means are combined.

Intervals are sample-based shortest contiguous intervals (HDIs), rather than
equal-tail credible intervals. The module also validates model coding,
respondent alignment, and observation consistency before posterior quantities
are combined or compared.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def posterior_interval(draws, hdi_prob=0.89, center="mean"):
    """Summarize posterior draws by their center and shortest contiguous HDI.

    The interval is estimated directly from the ordered posterior draws and is
    therefore not an equal-tail quantile interval.
    """
    if not 0 < hdi_prob < 1 or center not in {"mean", "median"}:
        raise ValueError("Use 0 < hdi_prob < 1 and center=mean or median.")
    values = np.asarray(draws, dtype=float).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Posterior draws must be nonempty and finite.")
    ordered = np.sort(values)
    span = int(np.floor(hdi_prob * len(ordered)))
    first = int(np.argmin(ordered[span:] - ordered[:len(ordered)-span]))
    middle = values.mean() if center == "mean" else np.median(values)
    return float(middle), float(ordered[first]), float(ordered[first+span])


def summarize_draws(draws, hdi_prob=0.89):
    """Return posterior mean and HDI bounds for a one-dimensional draw vector."""
    mean, low, high = posterior_interval(draws, hdi_prob)
    return {"mean": mean, "hdi_low": low, "hdi_high": high}


def find_posterior_name(attribute, level, available):
    prefixes = [f"{attribute}_"]
    if attribute.endswith("s"):
        prefixes.append(f"{attribute[:-1]}_")
    for prefix in prefixes:
        name = prefix + level
        if name in available:
            return name
    raise KeyError(f"Missing posterior coefficient for {attribute}: {level}")


def check_model_coding(idata, conjoint_config):
    actual = idata.posterior.attrs.get("coding")
    if actual is None:
        raise ValueError("Model lacks coding metadata. Do not guess: use a verified model/data pair.")
    if actual != conjoint_config["coding"]:
        raise ValueError(f"Model uses {actual} coding, config uses {conjoint_config['coding']}.")


def get_level_da(da, attribute, level, conjoint_config):
    """Return the posterior utility for one conjoint level.

    For non-baseline levels, the stored coefficient is returned directly.
    Under effect coding, the omitted baseline is reconstructed so that 
    coefficients within the attribute sum to zero.
    Under dummy coding, the baseline is fixed at zero.
    """
    spec = conjoint_config["attributes"][attribute]
    coding = conjoint_config["coding"]
    if coding not in {"effect", "dummy"}:
        raise ValueError("Unknown coding; expected effect or dummy.")
    if level not in spec["levels"]:
        raise KeyError(f"Unknown {attribute} level: {level}")
    available = set(map(str, da.coords["level"].values))
    nonbaseline = [x for x in spec["levels"] if x != spec["baseline"]]
    names = [find_posterior_name(attribute, x, available) for x in nonbaseline]
    if level != spec["baseline"]:
        return da.sel(level=find_posterior_name(attribute, level, available), drop=True)
    if coding == "dummy":
        return xr.zeros_like(da.sel(level=names[0], drop=True))
    return -da.sel(level=names).sum("level")


def get_level_draws(da, attribute, level, conjoint_config):
    result = get_level_da(da, attribute, level, conjoint_config)
    if set(result.dims) != {"chain", "draw"}:
        raise ValueError("Population summary needs chain/draw only; aggregate respondents first.")
    return result.transpose("chain", "draw").values.reshape(-1)


def build_level_summary_from_da(da, attr_order, levels_by_attr, conjoint_config, kind_label, hdi_prob=0.89):
    """Summarize posterior utilities for selected conjoint levels.

    Returns both a tidy summary table and the underlying posterior draws used
    by downstream plots."""   
    rows, draw_map = [], {}
    for attr in attr_order:
        for level in levels_by_attr[attr]:
            draws = get_level_draws(da, attr, level, conjoint_config)
            rows.append({
                "attribute": attr,
                "level": level,
                "kind": kind_label,
                **summarize_draws(draws, hdi_prob),
            })
            draw_map[(kind_label, attr, level)] = draws

    return pd.DataFrame(rows), draw_map



def quantity_da(idata, quantity, *, individual=False):
    """Return pre-event utility, event-related shift, or post-event utility.

    Post-event utility is always constructed draw by draw as pre + shift.
    With individual=True, respondent-specific posterior quantities are returned;
    otherwise population-level coefficients are used.
    """
    post = idata.posterior
    if individual:
        pre = post["partworth_individual"]
        if "shift_eff" in post:
            shift = post["shift_eff"]
        elif "shift" in post:
            shift = post["shift"].broadcast_like(pre)
        else:
            raise ValueError("This model has no individual shift.")
    else:
        pre = post["partworth_mean"]
        shift = post.get("shift_mean", post.get("shift"))
    if quantity == "pre":
        return pre
    if shift is None:
        raise ValueError("This model has no longitudinal shift.")
    if quantity == "shift":
        return shift
    if quantity == "post":
        return pre + shift
    raise ValueError("quantity must be pre, shift or post.")


def align_respondents(idata, metadata):
    """Align respondent metadata to the fitted model using verified identifiers.

    Respondents are matched by stored respondent IDs and provenance keys,
    never by row position. 
    """
    if idata.posterior.attrs.get("id_scheme") != "respondent_id_v2" or not hasattr(idata, "run_data"):
        raise ValueError("This .nc lacks verified respondent IDs. Refit with v2 or validate an explicit legacy mapping first.")
    if metadata["respondent_id"].duplicated().any() or metadata["respondent_id"].isna().any():
        raise ValueError("Respondent metadata must have unique nonmissing IDs.")
    ids = idata.run_data.coords["respondent"].values
    frame = metadata.set_index("respondent_id").reindex(ids)
    keys = idata.run_data["respondent_key"].values.astype(str)
    if frame["respondent_key"].isna().any() or not np.array_equal(frame["respondent_key"].astype(str).values, keys):
        raise ValueError("Model/metadata respondent mapping differs; do not join by row position.")
    if "respondent" in idata.posterior.dims and not np.array_equal(idata.posterior.respondent.values, ids):
        raise ValueError("Posterior and provenance respondent coordinates differ.")
    return frame.reset_index()


def group_summary(da, metadata, group_column, hdi_prob=0.89):
    """Summarize respondent-level posterior quantities by group.

    Respondent values are averaged within each group separately for every
    posterior draw. """
    if set(da.dims) != {"chain", "draw", "respondent"}:
        raise ValueError("Group summary expects one scalar per respondent/draw.")
    indexed = metadata.set_index("respondent_id").reindex(da.respondent.values)
    if indexed.index.duplicated().any():
        raise ValueError("Duplicate respondent in metadata.")
    rows = []
    for group, members in indexed.dropna(subset=[group_column]).groupby(group_column, sort=True):
        draws = da.sel(respondent=members.index.to_numpy()).mean("respondent").transpose("chain", "draw").values.ravel()
        rows.append({group_column: group, "n_respondents": len(members), **summarize_draws(draws, hdi_prob)})
    return pd.DataFrame(rows, columns=[group_column, "n_respondents", "mean", "hdi_low", "hdi_high"])


def individual_summary(da, hdi_prob=0.89):
    """Return posterior mean and HDI separately for each respondent."""
    values = da.transpose("respondent", "chain", "draw").values.reshape(da.sizes["respondent"], -1)
    return pd.DataFrame([{ "respondent_id": rid, **summarize_draws(draws, hdi_prob)}
                         for rid, draws in zip(da.respondent.values, values)])


def validate_model_comparison(models):
    """Require identical task/design/choice signatures for a standard LOO comparison."""
    signatures = [x.posterior.attrs.get("choice_signature") for x in models.values()]
    if not signatures or any(s is None for s in signatures) or len(set(signatures)) != 1:
        raise ValueError("LOO requires the same observed choices/design. Subsample robustness fits are not a common-observation ranking.")
    first = next(iter(models.values())).observed_data["choice"]
    for name, idata in models.items():
        if not hasattr(idata, "log_likelihood") or "choice" not in idata.log_likelihood:
            raise ValueError(f"{name}: missing log_likelihood.choice; fit with log_likelihood enabled.")
        if not np.array_equal(first.task, idata.observed_data["choice"].task) or not np.array_equal(first.values, idata.observed_data["choice"].values):
            raise ValueError("Observation order/values differ across models.")


def compute_level_surface(idata, attr, level, conjoint_config, nhv_axis, fv_axis, quantity="shift", pd_fixed=0.0, hdi_prob=0.89):
    """Evaluate the posterior utility surface over NHV and financial vulnerability.

    For one conjoint level, posterior utility is evaluated on every combination
    of the supplied NHV and FV grid values while psychological distance is held
    fixed. The returned matrices contain the posterior mean and HDI at each
    grid point.
    """
    post = idata.posterior
    needed = {"shift": ["shift_mean", "shift_nh", "shift_fin", "shift_psy"],
              "pre": ["partworth_mean", "gamma_nhv", "gamma_fv", "gamma_pd"]}
    if quantity not in {"pre", "shift", "post"}:
        raise ValueError("quantity must be pre, shift or post.")
    blocks = [needed["pre"], needed["shift"]] if quantity == "post" else [needed[quantity]]
    coefficients = [sum((get_level_draws(post[names[k]], attr, level, conjoint_config) for names in blocks)) for k in range(4)]
    mean = np.empty((len(fv_axis), len(nhv_axis))); low = np.empty_like(mean); high = np.empty_like(mean)
    for i, fv in enumerate(fv_axis):
        for j, nhv in enumerate(nhv_axis):
            draws = coefficients[0] + coefficients[1]*nhv + coefficients[2]*fv + coefficients[3]*pd_fixed
            mean[i, j], low[i, j], high[i, j] = posterior_interval(draws, hdi_prob)
    return mean, low, high


def _summarize_draw_matrix(matrix, prefix, hdi_prob=0.89):
    """Summarize a respondent-by-draw matrix with respondent-specific means and HDIs."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite respondent x posterior-draw matrix.")
    ordered = np.sort(values, axis=1)
    span = int(np.floor(hdi_prob * values.shape[1]))
    if span < 1 or span >= values.shape[1]:
        raise ValueError("Invalid hdi_prob for the available posterior draws.")
    widths = ordered[:, span:] - ordered[:, : values.shape[1] - span]
    first = np.argmin(widths, axis=1)
    rows = np.arange(values.shape[0])
    return pd.DataFrame({
        f"{prefix}_mean": values.mean(axis=1),
        f"{prefix}_hdi_low": ordered[rows, first],
        f"{prefix}_hdi_high": ordered[rows, first + span],
    })


def level_shift_decomposition(idata, attribute, level, conjoint_config, hdi_prob=0.89):
    """Decompose each respondent's event-related utility shift for one level.

    The total shift is separated draw by draw into the population-average shift
    and respondent-specific contributions from natural-hazard vulnerability,
    psychological distance, and financial vulnerability.
    """
    check_model_coding(idata, conjoint_config)
    posterior = idata.posterior
    needed = ["eta_nhv", "eta_pd", "shift_mean", "shift_nh", "shift_psy", "shift_fin"]
    missing = [name for name in needed if name not in posterior]
    if missing:
        raise ValueError(f"Respondent-impact plots require the full main HCM; missing {missing}.")
    if not hasattr(idata, "constant_data") or "fv_data" not in idata.constant_data:
        raise ValueError("Respondent-impact plots require fv_data in model constant_data.")

    respondent_ids = posterior.coords["respondent"].values
    eta_nhv = posterior["eta_nhv"].transpose("respondent", "chain", "draw").values
    eta_pd = posterior["eta_pd"].transpose("respondent", "chain", "draw").values
    eta_nhv = eta_nhv.reshape(len(respondent_ids), -1)
    eta_pd = eta_pd.reshape(len(respondent_ids), -1)

    fv_da = idata.constant_data["fv_data"]
    if "respondent" in fv_da.coords:
        fv_da = fv_da.sel(respondent=respondent_ids)
    fv = np.asarray(fv_da.values, dtype=float).reshape(-1)
    if len(fv) != len(respondent_ids):
        raise ValueError("fv_data does not align with posterior respondents.")

    delta = get_level_draws(posterior["shift_mean"], attribute, level, conjoint_config)
    phi_nhv = get_level_draws(posterior["shift_nh"], attribute, level, conjoint_config)
    phi_pd = get_level_draws(posterior["shift_psy"], attribute, level, conjoint_config)
    phi_fv = get_level_draws(posterior["shift_fin"], attribute, level, conjoint_config)
    n_draws = eta_nhv.shape[1]
    if not all(len(values) == n_draws for values in (delta, phi_nhv, phi_pd, phi_fv)):
        raise ValueError("Posterior coefficient draws do not align with latent respondent draws.")

    avg_shift = np.broadcast_to(delta, eta_nhv.shape)
    nhv_contrib = eta_nhv * phi_nhv[None, :]
    pd_contrib = eta_pd * phi_pd[None, :]
    fv_contrib = fv[:, None] * phi_fv[None, :]
    total_shift = avg_shift + nhv_contrib + pd_contrib + fv_contrib

    result = pd.DataFrame({"respondent_id": respondent_ids})
    for frame in (
        _summarize_draw_matrix(avg_shift, "avg_shift", hdi_prob),
        _summarize_draw_matrix(nhv_contrib, "nhv_contrib", hdi_prob),
        _summarize_draw_matrix(pd_contrib, "pd_contrib", hdi_prob),
        _summarize_draw_matrix(fv_contrib, "fv_contrib", hdi_prob),
        _summarize_draw_matrix(total_shift, "total_shift", hdi_prob),
    ):
        result = pd.concat([result, frame], axis=1)
    result["attribute"] = attribute
    result["level"] = level
    return result
