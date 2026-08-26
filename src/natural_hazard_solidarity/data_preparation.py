"""Data cleaning, wave linking, conjoint reshaping, coding, and sample filters."""

from __future__ import annotations

from math import ceil
import re

import numpy as np
import pandas as pd

from . import mappings as mp


def map_values(
    df: pd.DataFrame,
    mapping: dict,
    column_patterns: list[str] | None = None,
    numeric: bool = False,
) -> pd.DataFrame:
    """Replace values in columns selected by regex patterns."""
    out = df.copy()
    if column_patterns:
        regex = re.compile("|".join(column_patterns))
        columns = [c for c in out.columns if regex.search(c)]
    else:
        columns = out.columns.tolist()

    for col in columns:
        out[col] = out[col].replace(mapping)
        if numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().replace("", pd.NA)
    out.columns = out.columns.str.replace(".", "", regex=False)
    out.columns = out.columns.str.replace("\xa0", "", regex=False)
    out.columns = out.columns.str.replace(r"\s+", "", regex=True)
    out.columns = [c.replace("municipality", "benefits") for c in out.columns]
    if "Duration(inseconds)" in out.columns:
        out = out.rename(columns={"Duration(inseconds)": "duration"})
    for col in ["id", "m", "duration"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _as_finished(series: pd.Series) -> pd.Series:
    """Robustly interpret Qualtrics Finished values as booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "yes"})


def _remove_bad_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DistributionChannel" in out.columns:
        out = out[out["DistributionChannel"] != "preview"]
    if "Finished" in out.columns:
        out = out[_as_finished(out["Finished"])]
    if "Q_TerminateFlag" in out.columns:
        out = out[~out["Q_TerminateFlag"].isin(["PoorQuality", "NA", "QuotaMet", "Screened"])]
    return out.copy()


def _duration_invalid_mask(
    df: pd.DataFrame,
    lower_quantile: float | None,
    upper_quantile: float | None,
) -> pd.Series:
    if "duration" not in df.columns:
        return pd.Series(False, index=df.index)

    duration = pd.to_numeric(df["duration"], errors="coerce")
    valid = duration.dropna()
    if valid.empty:
        return pd.Series(False, index=df.index)

    lower = valid.quantile(lower_quantile) if lower_quantile is not None else -np.inf
    upper = valid.quantile(upper_quantile) if upper_quantile is not None else np.inf
    return duration.le(lower) | duration.gt(upper) | duration.isna()


def _straightliner_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    groups = {
        "costs": [c for c in df.columns if "likert_costs_" in c],
        "identity": [c for c in df.columns if "identity_group" in c],
        "gal_tan": [c for c in df.columns if "gal_tan_" in c],
        "deservingness": [c for c in df.columns if "deservingness" in c],
        "sensitivity": [c for c in df.columns if "sensitivity_nh_" in c],
        "psycho": [c for c in df.columns if "psycho_distance_" in c],
    }

    def is_straight(columns: list[str]) -> pd.Series:
        if not columns:
            return pd.Series(False, index=df.index)
        return df[columns].nunique(axis=1, dropna=True).eq(1)

    masks = {name: is_straight(columns) for name, columns in groups.items()}
    if mode == "deservingness":
        return masks["deservingness"]
    if mode == "medium":
        return masks["identity"] | masks["gal_tan"] | masks["deservingness"]
    if mode == "all":
        result = pd.Series(False, index=df.index)
        for mask in masks.values():
            result |= mask
        return result
    raise ValueError(f"Unknown straightliner mode: {mode}")


def _remove_shared_ips(df: pd.DataFrame, max_ids_per_ip: int) -> pd.DataFrame:
    if "IPAddress" not in df.columns or "id" not in df.columns:
        return df.copy()
    counts = df.groupby("IPAddress")["id"].nunique()
    bad_ips = counts[counts > max_ids_per_ip].index
    return df.loc[~df["IPAddress"].isin(bad_ips)].copy()


def _add_donation_dummies(df: pd.DataFrame) -> pd.DataFrame:
    if "solidarity_blatten" not in df.columns:
        return df.copy()
    out = df.copy()
    dummies = out["solidarity_blatten"].fillna("").str.get_dummies(sep=",")
    dummies.columns = dummies.columns.str.strip()
    dummies = dummies.reindex(columns=mp.DONATIONS, fill_value=0)
    dummies.columns = [
        "donated_blatten",
        "donated_other",
        "volunteered_blatten",
        "volunteered_other",
    ]
    return pd.concat([out, dummies], axis=1)


def _drop_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    pattern = "|".join(map(re.escape, mp.ANONYMIZE_COLS))
    columns = df.filter(regex=pattern).columns
    return df.drop(columns=columns, errors="ignore")


def clean_wave(df: pd.DataFrame, wave: str, config: dict) -> pd.DataFrame:
    """Clean one raw survey wave using the logic from prepare_data.ipynb."""
    out = _standardize_raw_columns(df)
    out = _remove_bad_quality(out)

    duration_config = config["duration"][wave]
    invalid_duration = _duration_invalid_mask(
        out,
        lower_quantile=duration_config.get("lower_quantile"),
        upper_quantile=duration_config.get("upper_quantile"),
    )
    straightliner = _straightliner_mask(out, config["quality"]["straightliner_mode"])

    attention_col = config["quality"]["attention_column"]
    attention_value = config["quality"]["attention_valid_value"]
    inattentive = (
        out[attention_col].ne(attention_value)
        if attention_col in out.columns
        else pd.Series(False, index=out.index)
    )
    out = out.loc[~(invalid_duration | straightliner | inattentive)].copy()

    if wave == "S0":
        numeric_columns = [c for c in mp.NUM_COLUMNS if c in out.columns]
        out[numeric_columns] = out[numeric_columns].apply(pd.to_numeric, errors="coerce")

    out = map_values(out, mp.LIKERT_MAP, mp.VALID_COLUMNS, numeric=True)
    out = map_values(out, mp.NH_EXPERIENCE_MAP, ["experience_nh"], numeric=True)

    if "finan_vulnerability_1" in out.columns:
        out["finan_vulnerability_1"] = 7 - pd.to_numeric(
            out["finan_vulnerability_1"], errors="coerce"
        )

    if wave == "S1":
        out = _add_donation_dummies(out)

    out = _remove_shared_ips(out, config["quality"]["max_ids_per_ip"])
    out = _drop_sensitive_columns(out)
    out = out.add_prefix(f"{wave}_")
    return out.reset_index(drop=True)


def merge_waves(
    s0: pd.DataFrame,
    s1: pd.DataFrame,
    keys: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Link cleaned S0 and S1 using the respondent-id mapping file."""
    map_keys = config["mapping_keys"]
    wave_keys = config["wave_keys"]
    keys = keys.copy()
    s0 = s0.copy()
    s1 = s1.copy()

    key_s0 = map_keys["S0"]
    key_s1 = map_keys["S1"]
    wave_s0 = f"S0_{wave_keys['S0']}"
    wave_s1 = f"S1_{wave_keys['S1']}"

    for frame, column in [(keys, key_s0), (keys, key_s1), (s0, wave_s0), (s1, wave_s1)]:
        if column not in frame.columns:
            raise KeyError(f"Expected linking column '{column}' was not found.")
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    merged = (
        keys.merge(s0, how="inner", left_on=key_s0, right_on=wave_s0)
        .merge(s1, how="inner", left_on=key_s1, right_on=wave_s1)
        .drop(columns=["S0_id", "S0_m", "S1_id", "S1_m"], errors="ignore")
    )
    merged = merged.drop_duplicates(subset=[key_s0, key_s1], keep="first").reset_index(drop=True)
    merged["respondent_id"] = np.arange(1, len(merged) + 1)

    merged = map_values(
        merged,
        mp.DEMOGRAPHICS_DICT,
        [
            r"_gender$",
            r"_age$",
            r"_education$",
            r"_income$",
            r"_language$",
            r"_language_region$",
            r"_party_choice$",
        ],
    )
    merged = map_values(
        merged,
        mp.TRANSLATION_DICT,
        [r"^S._choice._exemptions", r"^S._choice._costs", r"^S._choice._benefits"],
    )
    merged = map_values(merged, mp.ARTICLES_MAP, [r"^S1_media_2"])
    merged = map_values(merged, mp.CONVERSATIONS_MAP, [r"^S1_Q50"])
    return merged


def _complete_construct_wave(
    df: pd.DataFrame,
    columns: list[str],
    min_items: int,
    method: str,
    lo: int,
    hi: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Complete one construct within one wave if enough items are observed."""

    values = df[columns].apply(pd.to_numeric, errors="coerce")

    # Keep only valid Likert values.
    values = values.where(values.apply(lambda series: series.between(lo, hi)))
    n_observed = values.notna().sum(axis=1)
    usable = n_observed >= min_items

    if method == "median":
        center = values.median(axis=1)
    elif method == "mean":
        center = values.mean(axis=1)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    center = np.floor(center + 0.5).clip(lo, hi)

    # Only impute respondents for whom the construct is sufficiently observed.
    values.loc[usable] = (values.loc[usable].T.fillna(center.loc[usable]).T)
    return values, usable

def _complete_construct_two_waves(
    df: pd.DataFrame,
    items: list[str],
    min_items: int,
    method: str,
    lo: int,
    hi: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Complete a construct within waves and fall back across waves."""
    out = df.copy()

    s0_cols = [f"S0_{item}" for item in items]
    s1_cols = [f"S1_{item}" for item in items]

    missing = [col for col in s0_cols + s1_cols if col not in out.columns]
    if missing:
        raise KeyError(f"Missing construct columns: {missing}")

    s0_values, s0_usable = _complete_construct_wave(out, s0_cols, min_items=min_items, method=method, lo=lo, hi=hi,)
    s1_values, s1_usable = _complete_construct_wave(out, s1_cols, min_items=min_items, method=method, lo=lo, hi=hi,)

    # S0 insufficient, but S1 usable: use the completed S1 construct for S0.
    use_s1_for_s0 = ~s0_usable & s1_usable
    s0_values.loc[use_s1_for_s0] = (s1_values.loc[use_s1_for_s0].to_numpy())
    # S1 insufficient, but S0 usable: use the completed S0 construct for S1.
    use_s0_for_s1 = ~s1_usable & s0_usable
    s1_values.loc[use_s0_for_s1] = (s0_values.loc[use_s0_for_s1].to_numpy())

    out[s0_cols] = s0_values
    out[s1_cols] = s1_values

    # Respondent survives if at least one wave provides a usable version of the construct.
    keep = s0_usable | s1_usable
    return out, keep

def _complete_single_item_two_waves(
    df: pd.DataFrame,
    item: str,
    lo: int,
    hi: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Complete a single-item variable across S0 and S1.
    If one wave is missing, use the other wave. If both waves are missing, exclude the respondent.
    """
    out = df.copy()

    s0_col = f"S0_{item}"
    s1_col = f"S1_{item}"

    if s0_col not in out.columns or s1_col not in out.columns:
        raise KeyError(f"Expected columns '{s0_col}' and '{s1_col}'.")

    s0 = pd.to_numeric(out[s0_col], errors="coerce")
    s1 = pd.to_numeric(out[s1_col], errors="coerce")
    s0 = s0.where(s0.between(lo, hi))
    s1 = s1.where(s1.between(lo, hi))
    out[s0_col] = s0.fillna(s1)
    out[s1_col] = s1.fillna(s0)

    keep = (out[s0_col].notna() & out[s1_col].notna())
    return out, keep

def add_acceptance_consistency(df: pd.DataFrame, max_inconsistent: int) -> pd.DataFrame:
    """Reproduce the conjoint preference/acceptance consistency check from the notebook."""
    out = df.copy()
    preference_columns = [c for c in out.columns if c.endswith("_conjoint_prefer")]
    consistency_columns: list[str] = []

    for preference_col in preference_columns:
        prefix = preference_col.removesuffix("_conjoint_prefer")
        acceptance_1 = f"{prefix}_conjoint_acceptance_1"
        acceptance_2 = f"{prefix}_conjoint_acceptance_2"
        if acceptance_1 not in out.columns or acceptance_2 not in out.columns:
            continue

        for column in [acceptance_1, acceptance_2]:
            out[column] = out[column].replace(mp.LIKERT_MAP)
            out[column] = pd.to_numeric(out[column], errors="coerce")

        check_col = f"{prefix}_acceptance_consistent"
        out[check_col] = np.select(
            [
                out[preference_col].eq("Option 1"),
                out[preference_col].eq("Option 2"),
            ],
            [
                out[acceptance_1].ge(out[acceptance_2]),
                out[acceptance_2].ge(out[acceptance_1]),
            ],
            default=False,
        ).astype(bool)
        consistency_columns.append(check_col)

    if consistency_columns:
        out["n_acceptance_inconsistent"] = (~out[consistency_columns]).sum(axis=1)
        # The old notebook called this "all_acceptance_consistent" but allowed up to
        # two inconsistent tasks. This threshold is now explicit in config.yaml.
        out["all_acceptance_consistent"] = out["n_acceptance_inconsistent"].le(max_inconsistent)
    else:
        out["n_acceptance_inconsistent"] = pd.NA
        out["all_acceptance_consistent"] = False
    return out


def prepare_analysis_sample(
    df: pd.DataFrame,
    item_config: dict,
    imputation_config: dict,
    acceptance_config: dict,
) -> pd.DataFrame:
    """
    Prepare the respondent sample used by the HCM.

    Missing-data logic:
    - Multi-item constructs require at least 2 observed items.
    - Missing third items are completed within the same wave.
    - If a wave is insufficient, use the construct from the other wave.
    - Respondents are removed only if a required construct cannot be
      formed from either wave.
    """
    out = df.copy()
    method = imputation_config["method"]
    lo = imputation_config["likert_min"]
    hi = imputation_config["likert_max"]

    # Natural-hazard vulnerability
    nh_items = item_config["sensitivity_nh"]
    nh_min_items = imputation_config["min_items"]["sensitivity_nh"]
    out, keep_nh = _complete_construct_two_waves(out, items=nh_items, min_items=nh_min_items, method=method, lo=lo, hi=hi,)

    # Financial vulnerability
    fv_items = item_config["financial_vulnerability"]
    if len(fv_items) != 1:
        raise ValueError("financial_vulnerability is currently expected to contain exactly one item.")

    out, keep_fv = _complete_single_item_two_waves(out, item=fv_items[0], lo=lo, hi=hi,)

    # Psychological distance
    # Currently this construct is only needed from S1. Require at least 2 of the 3 items.
    pd_items = item_config["psychological_distance"]
    pd_min_items = imputation_config["min_items"]["psychological_distance"]
    
    pd_columns = [f"S1_{item}" for item in pd_items]
    missing_pd_columns = [col for col in pd_columns if col not in out.columns]
    if missing_pd_columns:
        raise KeyError(f"Missing psychological-distance columns: " f"{missing_pd_columns}")

    pd_values, keep_pd = _complete_construct_wave(out, columns=pd_columns, min_items=pd_min_items, method=method, lo=lo, hi=hi,)
    out[pd_columns] = pd_values

    # Final HCM sample
    keep = keep_nh & keep_fv & keep_pd
    out = out.loc[keep].copy()

    # Conjoint acceptance consistency
    out = add_acceptance_consistency(out, max_inconsistent=acceptance_config["max_inconsistent_tasks"],)

    # Blatten climate-change indicator
    if "S1_climate_blatten" in out.columns:
        out["blatten_cc"] = (
            out["S1_climate_blatten"]
            .astype("string")
            .str.contains("climate change", case=False,na=False,)
        )
    return out.reset_index(drop=True)

def _respondent_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    return [column for column in requested if column in df.columns]


def build_conjoint_long(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Transform the two-wave wide survey into one row per conjoint alternative."""
    attribute_pattern = r"^S\d+_choice\d+_(costs|benefits|exemptions)\d+$"
    choice_pattern = r"^S\d+_\d+_conjoint_prefer$"
    attribute_columns = df.filter(regex=attribute_pattern).columns.tolist()
    choice_columns = df.filter(regex=choice_pattern).columns.tolist()
    if not attribute_columns:
        raise ValueError("No conjoint attribute columns were found.")
    if not choice_columns:
        raise ValueError("No conjoint preference columns were found.")

    respondent_columns = _respondent_columns(df, config["respondent_columns"])
    if "respondent_id" not in respondent_columns:
        respondent_columns = ["respondent_id"] + respondent_columns

    attributes = df.melt(
        id_vars=respondent_columns,
        value_vars=attribute_columns,
        var_name="var",
        value_name="value",
    )
    attributes[["nh_event", "task", "attribute", "option"]] = attributes["var"].str.extract(
        r"S(\d+)_choice(\d+)_(costs|benefits|exemptions)(\d+)"
    )
    for column in ["nh_event", "task", "option"]:
        attributes[column] = attributes[column].astype(int)

    options = (
        attributes.pivot(
            index=respondent_columns + ["nh_event", "task", "option"],
            columns="attribute",
            values="value",
        )
        .reset_index()
        .copy()
    )
    options.columns.name = None

    choices = df.melt(
        id_vars=["respondent_id"],
        value_vars=choice_columns,
        var_name="preference_var",
        value_name="choice",
    )
    choices["choice"] = choices["choice"].replace(mp.PREFERENCE_MAP)
    choices["choice"] = pd.to_numeric(choices["choice"], errors="coerce")
    choices[["nh_event", "task"]] = choices["preference_var"].str.extract(
        r"S(\d+)_(\d+)_conjoint_prefer"
    )
    choices["nh_event"] = choices["nh_event"].astype(int)
    choices["task"] = choices["task"].astype(int)
    choices = choices[["respondent_id", "nh_event", "task", "choice"]]

    out = options.merge(choices, on=["respondent_id", "nh_event", "task"], how="left")
    out["chosen"] = out["choice"].eq(out["option"]).astype("int8")
    out = out.drop(columns="choice")
    out = out.sort_values(["respondent_id", "nh_event", "task", "option"]).reset_index(drop=True)
    out["task_id"] = out.groupby(["respondent_id", "nh_event", "task"], sort=False).ngroup()

    task_sizes = out.groupby("task_id").size()
    if not task_sizes.eq(2).all():
        bad = task_sizes[task_sizes.ne(2)].head().to_dict()
        raise ValueError(f"Some conjoint tasks do not contain exactly two alternatives: {bad}")
    chosen_counts = out.groupby("task_id")["chosen"].sum()
    if not chosen_counts.eq(1).all():
        bad = chosen_counts[chosen_counts.ne(1)].head().to_dict()
        raise ValueError(f"Some conjoint tasks do not contain exactly one chosen alternative: {bad}")
    return out


def encode_conjoint(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add effect-coded or reference-dummy-coded design columns."""
    coding = config["coding"]
    if coding not in {"dummy", "effect"}:
        raise ValueError("conjoint.coding must be 'dummy' or 'effect'.")

    out = df.copy()
    blocks: list[pd.DataFrame] = []
    for attribute, specification in config["attributes"].items():
        baseline = specification["baseline"]
        levels = specification["levels"]
        nonbaseline = [level for level in levels if level != baseline]

        observed = set(out[attribute].dropna().unique())
        unknown = observed.difference(levels)
        if unknown:
            raise ValueError(f"Unknown levels in {attribute}: {sorted(unknown)}")

        block = pd.DataFrame(index=out.index)
        for level in nonbaseline:
            # Keep the old naming convention so later posterior code remains easy to migrate.
            column = f"{attribute}_{level}"
            if coding == "dummy":
                block[column] = out[attribute].eq(level).astype("int8")
            else:
                block[column] = np.select(
                    [out[attribute].eq(level), out[attribute].eq(baseline)],
                    [1, -1],
                    default=0,
                ).astype("int8")
        blocks.append(block)

    return pd.concat([out] + blocks, axis=1)


def design_columns(conjoint_config: dict) -> list[str]:
    """Return encoded feature names in a stable, config-defined order."""
    columns: list[str] = []
    for attribute, specification in conjoint_config["attributes"].items():
        baseline = specification["baseline"]
        columns.extend(
            f"{attribute}_{level}"
            for level in specification["levels"]
            if level != baseline
        )
    return columns


def filter_model_sample(df: pd.DataFrame, sample_config: dict) -> pd.DataFrame:
    """Apply a respondent or task-level robustness-sample definition."""
    sample_type = sample_config["type"]
    out = df.copy()

    if sample_type == "full":
        return out

    if sample_type == "filter":
        column = sample_config["column"]
        if column not in out.columns:
            raise KeyError(f"Sample filter column '{column}' was not found in the conjoint data.")
        operator = sample_config["operator"]
        value = sample_config["value"]
        if operator == "equals":
            return out.loc[out[column].eq(value)].copy()
        if operator == "not_equals":
            return out.loc[out[column].ne(value)].copy()
        if operator == "is_true":
            return out.loc[out[column].fillna(False).astype(bool)].copy()
        raise ValueError(f"Unknown sample filter operator: {operator}")

    if sample_type == "exclude_tasks":
        conditions = sample_config["conditions"]
        match = pd.Series(True, index=out.index)
        for column, value in conditions.items():
            if column not in out.columns:
                raise KeyError(f"Task-exclusion column '{column}' was not found.")
            match &= out[column].eq(value)
        task_ids = out.loc[match, "task_id"].unique()
        return out.loc[~out["task_id"].isin(task_ids)].copy()

    raise ValueError(f"Unknown sample type: {sample_type}")
