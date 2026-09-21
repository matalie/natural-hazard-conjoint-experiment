"""Robustness-comparison plotting and validation.

Scientific logic migrated from ``analyze_results.ipynb`` cells 56-76.
The module is Snakemake-agnostic: callers provide already loaded InferenceData
objects and, optionally, the conjoint frame used to validate robustness samples.
"""
from __future__ import annotations

from collections.abc import Mapping

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .data_preparation import filter_model_sample
from .posterior import (check_model_coding, get_level_draws, quantity_da, summarize_draws)
from .plotting import (LEVEL_ORDER, ATTRIBUTE_ORDER, plot_forest_panels, PURPLE, DARK_GREY)

def _model_n_respondents(idata) -> int:
    value = idata.posterior.attrs.get("n_respondents")
    if value is not None:
        return int(value)
    return int(idata.posterior.sizes.get("respondent", 0))


def _model_n_tasks(idata) -> int:
    value = idata.posterior.attrs.get("n_tasks")
    if value is not None:
        return int(value)
    if hasattr(idata, "observed_data") and "choice" in idata.observed_data:
        return int(idata.observed_data["choice"].sizes.get("task", 0))
    return 0



def _normalised_ids(values) -> tuple[str, ...]:
    return tuple(map(str, np.asarray(values).reshape(-1)))


def _model_respondent_ids(idata):
    if "respondent" in idata.posterior.coords:
        return _normalised_ids(idata.posterior.coords["respondent"].values)
    return None


def _model_task_ids(idata):
    if hasattr(idata, "observed_data") and "choice" in idata.observed_data:
        choice = idata.observed_data["choice"]
        if "task" in choice.coords:
            return _normalised_ids(choice.coords["task"].values)
    if "task" in idata.posterior.coords:
        return _normalised_ids(idata.posterior.coords["task"].values)
    return None

def build_sample_audit(
    models: Mapping[str, object],
    specification: Mapping,
    run_configs: Mapping | None = None,
    sample_configs: Mapping | None = None,
    conjoint_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Verify that each robustness NetCDF contains the configured sample.

    In addition to respondent/task counts, the audit compares the exact
    respondent and task identifiers stored in the fitted NetCDF against the
    sample reconstructed from the current conjoint parquet and config. This
    catches stale or accidentally mis-filtered subsample fits even when their
    sample sizes happen to be similar.
    """
    rows = []
    expected_n = specification.get("expected_n_respondents", {}) or {}
    expected_tasks_cfg = specification.get("expected_n_tasks", {}) or {}

    for run_name in specification["runs"]:
        if run_name not in models:
            raise KeyError(f"Missing robustness model: {run_name}")
        idata = models[run_name]
        attrs = idata.posterior.attrs
        stored_run = attrs.get("run_name")
        stored_sample = attrs.get("sample")
        got_n = _model_n_respondents(idata)
        got_tasks = _model_n_tasks(idata)
        got_respondent_ids = _model_respondent_ids(idata)
        got_task_ids = _model_task_ids(idata)

        configured_sample = None
        expected_rows = None
        expected_respondents = None
        expected_tasks = None
        respondent_ids_match = None
        task_ids_match = None
        missing_respondents = 0
        unexpected_respondents = 0
        missing_tasks = 0
        unexpected_tasks = 0
        filter_description = ""
        problems = []

        if stored_run is not None and str(stored_run) != str(run_name):
            problems.append(f"NetCDF run_name={stored_run!r}")

        if run_configs and run_name in run_configs:
            configured_sample = run_configs[run_name].get("sample")
            if (
                stored_sample is not None
                and configured_sample is not None
                and str(stored_sample) != str(configured_sample)
            ):
                problems.append(
                    f"NetCDF sample={stored_sample!r}, config sample={configured_sample!r}"
                )

        if (
            configured_sample is not None
            and sample_configs
            and configured_sample in sample_configs
            and conjoint_frame is not None
        ):
            sample_cfg = sample_configs[configured_sample]
            expected_frame = filter_model_sample(conjoint_frame, sample_cfg)
            expected_rows = int(len(expected_frame))
            expected_respondents = int(expected_frame["respondent_id"].nunique())
            expected_tasks = int(expected_frame["task_id"].nunique())
            filter_description = repr(dict(sample_cfg))

            if got_n != expected_respondents:
                problems.append(
                    f"respondents in NetCDF={got_n}, configured sample={expected_respondents}"
                )
            if got_tasks and got_tasks != expected_tasks:
                problems.append(
                    f"tasks in NetCDF={got_tasks}, configured sample={expected_tasks}"
                )

            expected_resp_set = set(_normalised_ids(expected_frame["respondent_id"].drop_duplicates().values))
            expected_task_set = set(_normalised_ids(expected_frame["task_id"].drop_duplicates().values))

            if got_respondent_ids is not None:
                got_resp_set = set(got_respondent_ids)
                respondent_ids_match = got_resp_set == expected_resp_set
                missing_respondents = len(expected_resp_set - got_resp_set)
                unexpected_respondents = len(got_resp_set - expected_resp_set)
                if not respondent_ids_match:
                    problems.append(
                        "respondent IDs differ "
                        f"(missing={missing_respondents}, unexpected={unexpected_respondents})"
                    )
            else:
                problems.append("NetCDF has no verifiable respondent coordinate")

            if got_task_ids is not None:
                got_task_set = set(got_task_ids)
                task_ids_match = got_task_set == expected_task_set
                missing_tasks = len(expected_task_set - got_task_set)
                unexpected_tasks = len(got_task_set - expected_task_set)
                if not task_ids_match:
                    problems.append(
                        "task IDs differ "
                        f"(missing={missing_tasks}, unexpected={unexpected_tasks})"
                    )
            else:
                problems.append("NetCDF has no verifiable task coordinate")

        if run_name in expected_n and got_n != int(expected_n[run_name]):
            problems.append(
                f"respondents in NetCDF={got_n}, expected_n_respondents={int(expected_n[run_name])}"
            )
        if run_name in expected_tasks_cfg and got_tasks != int(expected_tasks_cfg[run_name]):
            problems.append(
                f"tasks in NetCDF={got_tasks}, expected_n_tasks={int(expected_tasks_cfg[run_name])}"
            )

        rows.append(
            {
                "run": run_name,
                "configured_sample": configured_sample,
                "stored_sample": stored_sample,
                "netcdf_respondents": got_n,
                "netcdf_tasks": got_tasks,
                "configured_rows": expected_rows,
                "configured_respondents": expected_respondents,
                "configured_tasks": expected_tasks,
                "respondent_ids_match": respondent_ids_match,
                "task_ids_match": task_ids_match,
                "missing_respondents": missing_respondents,
                "unexpected_respondents": unexpected_respondents,
                "missing_tasks": missing_tasks,
                "unexpected_tasks": unexpected_tasks,
                "filter": filter_description,
                "status": "mismatch" if problems else "ok",
                "problem": "; ".join(problems),
            }
        )

    return pd.DataFrame(rows)

def assert_sample_audit(audit: pd.DataFrame) -> None:
    bad = audit.loc[audit["status"].ne("ok")]
    if bad.empty:
        return
    detail = "\n".join(
        f"- {row.run}: {row.problem}" for row in bad.itertuples(index=False)
    )
    raise ValueError(
        "Robustness model/sample mismatch. The plot was stopped rather than "
        "silently comparing the wrong subsample:\n" + detail
    )


def resolve_plot_levels(specification: Mapping, conjoint_config: Mapping) -> dict[str, list[str]]:
    """Resolve which conjoint levels are shown in a robustness comparison.

    ``levels`` may be omitted/``all`` or may specify a subset per attribute.
    Attributes not mentioned in a mapping keep all configured levels. An
    optional ``exclude_levels`` mapping is applied afterwards.
    """
    configured = {
        attr: [level for level in LEVEL_ORDER[attr] if level in conjoint_config["attributes"][attr]["levels"]]
        for attr in ATTRIBUTE_ORDER
    }
    requested = specification.get("levels", "all")

    if requested in (None, "all"):
        selected = {attr: list(levels) for attr, levels in configured.items()}
    elif isinstance(requested, Mapping):
        selected = {}
        for attr in ATTRIBUTE_ORDER:
            value = requested.get(attr, "all")
            if value in (None, "all"):
                levels = list(configured[attr])
            else:
                if isinstance(value, str):
                    value = [value]
                levels = list(value)
                unknown = [level for level in levels if level not in configured[attr]]
                if unknown:
                    raise ValueError(
                        f"Unknown robustness level(s) for {attr}: {unknown}. "
                        f"Valid levels are {configured[attr]}."
                    )
            if levels:
                selected[attr] = levels
    else:
        raise ValueError("robustness 'levels' must be 'all' or attribute -> level-list mapping")

    excluded = specification.get("exclude_levels", {}) or {}
    for attr, values in excluded.items():
        if attr not in configured:
            raise ValueError(f"Unknown robustness attribute in exclude_levels: {attr}")
        if isinstance(values, str):
            values = [values]
        unknown = [level for level in values if level not in configured[attr]]
        if unknown:
            raise ValueError(f"Unknown excluded level(s) for {attr}: {unknown}")
        if attr in selected:
            selected[attr] = [level for level in selected[attr] if level not in set(values)]
            if not selected[attr]:
                selected.pop(attr)

    if not selected:
        raise ValueError("Robustness plot has no levels to display.")
    return selected


def _summary_for_selected_levels(
    da,
    selected_levels: Mapping[str, list[str]],
    conjoint_config: Mapping,
    kind_label: str,
    hdi_prob: float,
    recenter: bool,
):
    """Summarize selected levels, optionally recentering within shown levels.

    The recentering reproduces the approach from ``analyze_results.ipynb``
    cells 74-76 for comparisons where one level is deliberately omitted.
    """
    rows = []
    draws_map = {}

    for attr, levels in selected_levels.items():
        raw = [get_level_draws(da, attr, level, conjoint_config) for level in levels]
        matrix = np.vstack(raw)
        if recenter and len(levels) > 1:
            matrix = matrix - matrix.mean(axis=0, keepdims=True)

        for level, draws in zip(levels, matrix):
            rows.append(
                {
                    "attribute": attr,
                    "level": level,
                    "kind": kind_label,
                    **summarize_draws(draws, hdi_prob),
                }
            )
            draws_map[(kind_label, attr, level)] = draws

    return pd.DataFrame(rows), draws_map


def plot_robustness(
    models: Mapping[str, object],
    specification: Mapping,
    conjoint_config: Mapping,
    hdi_prob: float = 0.89,
):
    """
    Create a legacy-style pre/shift/post robustness overlay.
    Returns ``(figure, summary_table)``. File writing belongs to the workflow wrapper.
    """
    run_names = list(specification["runs"])
    colours = list(specification.get("colors", [DARK_GREY, DARK_GREY, PURPLE]))
    alphas = list(specification.get("alphas", [1.0, 0.35, 0.8]))
    if len(colours) < len(run_names) or len(alphas) < len(run_names):
        raise ValueError("Provide one color and alpha per robustness run.")

    selected_levels = resolve_plot_levels(specification, conjoint_config)
    attr_order = [attr for attr in ATTRIBUTE_ORDER if attr in selected_levels]
    quantities = list(specification.get("quantities", ["pre", "shift"]))
    recenter = bool(specification.get("recenter_selected_levels", False))

    frames = []
    draws = {}
    colour_map = {}
    handles = []

    for i, name in enumerate(run_names):
        idata = models[name]
        check_model_coding(idata, conjoint_config)
        n = _model_n_respondents(idata)
        label = specification.get("labels", {}).get(name, name) + f" (n={n})"
        handles.append(
            Line2D(
                [0], [0], color=colours[i], alpha=alphas[i], marker="o",
                linewidth=2, label=label,
            )
        )

        for quantity in quantities:
            kind = f"{name}:{quantity}"
            frame, values = _summary_for_selected_levels(
                da=quantity_da(idata, quantity),
                selected_levels=selected_levels,
                conjoint_config=conjoint_config,
                kind_label=kind,
                hdi_prob=hdi_prob,
                recenter=recenter,
            )
            frame["run"] = name
            frame["quantity"] = quantity
            frame["recentered"] = recenter
            frames.append(frame)
            draws.update(values)
            colour_map.update({(attr, kind): colours[i] for attr in attr_order})

    table = pd.concat(frames, ignore_index=True)
    low = min(0.0, float(table.hdi_low.min()))
    high = max(0.0, float(table.hdi_high.max()))
    pad = max(0.05 * (high - low), 0.02)
    limits = tuple(specification.get("xlim", [low - pad, high + pad]))

    xlabel = {
        "pre": "Pre-event average utility",
        "shift": "Average utility shift",
        "post": "Post-event average utility",
    }
    panels = [
        {
            "kinds": [f"{name}:{quantity}" for name in run_names],
            "alphas": alphas[: len(run_names)],
            "show_density": False,
            "xlim": limits,
            "xlabel": xlabel[quantity],
            "zero": True,
        }
        for quantity in quantities
    ]

    fig, _ = plot_forest_panels(
        plot_df=table,
        attr_order=attr_order,
        levels_by_attr=selected_levels,
        panels=panels,
        subplot_color_map=colour_map,
        draws_map=draws,
        fig_width=float(specification.get("fig_width", 11.4)),
        bottom_margin=float(specification.get("bottom_margin", 0.21)),
        level_wrap_width=int(specification.get("level_wrap_width", 41)),
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.65, 0.01),
        frameon=False,
        ncol=int(specification.get("legend_ncol", 1)),
    )
    return fig, table
