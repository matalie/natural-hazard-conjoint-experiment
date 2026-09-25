"""Survey data preparation inlcuding cleaning, item-wise imputation, choice design and sample filters.

"""
from __future__ import annotations
import re
import hashlib
import numpy as np
import pandas as pd
from . import mappings as mp

def _remove_bad_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude respondents with bad finishing flags"""
    required = ["DistributionChannel", "Finished", "Q_TerminateFlag"]
    missing = [c for c in required if c not in df]
    if missing:
        raise KeyError(f"Missing quality-control columns: {missing}")
    mask = df["DistributionChannel"].ne("preview") & df["Finished"].ne(False)
    mask &= ~df["Q_TerminateFlag"].isin(["PoorQuality", "NA", "QuotaMet", "Screened"])
    return df.loc[mask].copy()

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
    if out.columns.duplicated().any():
        raise ValueError("Column normalization produced duplicate names.")
    out.columns = [c.replace("municipality", "benefits") for c in out.columns]
    if "Duration(inseconds)" in out.columns:
        out = out.rename(columns={"Duration(inseconds)": "duration"})
    for col in ["id", "m", "duration"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def _duration_invalid_mask(df: pd.DataFrame, lower_quantile: float | None, upper_quantile: float | None,) -> pd.Series:
    if "duration" not in df.columns:
        raise KeyError("Missing survey duration column.")

    duration = pd.to_numeric(df["duration"], errors="coerce")
    valid = duration.dropna()
    if valid.empty:
        return pd.Series(True, index=df.index)

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
    """Clean one raw survey wave"""
    out = _standardize_raw_columns(df)
    out = _remove_bad_quality(out)

    # Quality exclusions are applied before item recoding and wave merging.
    duration_config = config["duration"][wave]
    invalid_duration = _duration_invalid_mask(
        out,
        lower_quantile=duration_config.get("lower_quantile"),
        upper_quantile=duration_config.get("upper_quantile"),
    )
    straightliner = _straightliner_mask(out, config["quality"]["straightliner_mode"])

    attention_col = config["quality"]["attention_column"]
    attention_value = config["quality"]["attention_valid_value"]
    if attention_col not in out:
        raise KeyError(f"Missing attention-check column: {attention_col}")
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

    # Reverse-code financial vulnerability so that larger values consistently represent greater financial vulnerability.
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

    keys = keys.dropna(subset=[key_s0, key_s1])
    s0 = s0.dropna(subset=[wave_s0])
    s1 = s1.dropna(subset=[wave_s1])

    merged = (
        keys.merge(s0, how="inner", left_on=key_s0, right_on=wave_s0)
        .merge(s1, how="inner", left_on=key_s1, right_on=wave_s1)
        .drop(columns=["S0_id", "S0_m", "S1_id", "S1_m"], errors="ignore")
    )
    merged = merged.drop_duplicates(subset=[key_s0, key_s1], keep="first").reset_index(drop=True)
    id_col = config.get("respondent_id_column")
    if id_col:
        if id_col not in merged or merged[id_col].isna().any() or merged[id_col].duplicated().any():
            raise ValueError("Configured respondent_id_column must be complete and unique.")
        merged["respondent_id"] = merged[id_col].astype(str)
    else:
        merged["respondent_id"] = np.arange(1, len(merged) + 1)

    merged["respondent_key"] = [hashlib.sha256(f"{a}|{b}".encode()).hexdigest()
                                for a, b in zip(merged[key_s0], merged[key_s1])]
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
                out[preference_col].eq("Option 1").fillna(False).to_numpy(dtype=bool),
                out[preference_col].eq("Option 2").fillna(False).to_numpy(dtype=bool),
            ],
            [
                out[acceptance_1].ge(out[acceptance_2]).fillna(False).to_numpy(dtype=bool),
                out[acceptance_2].ge(out[acceptance_1]).fillna(False).to_numpy(dtype=bool),
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
        value = sample_config.get("value")
        if operator == "equals":
            return out.loc[out[column].eq(value)].copy()
        if operator == "not_equals":
            return out.loc[out[column].ne(value).fillna(True)].copy()
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


def _fill_within_wave(df, items, prefix, min_items, method="median", lo=1, hi=6):
    """Impute latent factor item values using remaining factor items if <= 1/3 ist missing"""
    columns = [prefix + item for item in items]
    if not 1 <= min_items <= len(columns):
        raise ValueError("min_items must be between 1 and the number of items.")
    out = df.copy()
    values = out[columns].apply(pd.to_numeric, errors="coerce")
    values = values.where((values >= lo) & (values <= hi))
    eligible = values.notna().sum(axis=1).ge(min_items)
    if method not in {"median", "mean"}:
        raise ValueError("Imputation method must be median or mean.")
    center = getattr(values, method)(axis=1)
    center = np.floor(center + 0.5).clip(lo, hi)
    values.loc[eligible] = values.loc[eligible].T.fillna(center.loc[eligible]).T
    out[columns] = values
    return out


def _fallback_between_waves(df, items, target, source, lo=1, hi=6):
    """Fallback_fill: fill missing latent construct items of respondents between waves in case <= 1/3 is missing"""
    out = df.copy()
    for item in items:
        target_col, source_col = target + item, source + item
        source_values = pd.to_numeric(out[source_col], errors="coerce")
        source_values = source_values.where(source_values.between(lo, hi))
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(source_values)
    return out


def prepare_analysis_sample(df, item_config, imputation_config, acceptance_config, *, return_audit=False):
    """Prepare the respondent-level analysis sample.

    Missing NHV and financial-vulnerability items are first imputed within
    construct and wave and, if still missing, from the corresponding item in
    the other survey wave. Psychological-distance items are imputed within S1
    only. Respondents with incomplete or invalid construct information after
    imputation are excluded.

    The function also derives a conjoint response-consistency indicator by
    comparing each stated preference with the acceptance ratings of the two
    alternatives.
    """
    out = df.copy()
    lo, hi = imputation_config["likert_min"], imputation_config["likert_max"]
    method = imputation_config["method"]
    minimum = imputation_config["min_items"]
    groups = {"sensitivity_nh": item_config["sensitivity_nh"],
              "financial_vulnerability": item_config["financial_vulnerability"]}
    main_items = [item for items in groups.values() for item in items]
    pd_items = item_config["psychological_distance"]
    required = ([prefix + item for prefix in ["S0_", "S1_"] for item in main_items]
                + ["S1_" + item for item in pd_items])
    missing = [c for c in required if c not in out]
    if missing:
        raise KeyError(f"Missing required construct columns: {missing}")
    if out["respondent_id"].isna().any() or out["respondent_id"].duplicated().any():
        raise ValueError("Analysis sample needs exactly one row per respondent_id.")

    original = out[required].apply(pd.to_numeric, errors="coerce")
    original = original.where((original >= lo) & (original <= hi))
    for prefix in ["S0_", "S1_"]:
        for name, items in groups.items():
            threshold = minimum.get(name, 1 if len(items) == 1 else 2)
            out = _fill_within_wave(out, items, prefix, threshold, method, lo, hi)
    within = out[required].copy()
    out = _fallback_between_waves(out, main_items, "S0_", "S1_", lo, hi)
    out = _fallback_between_waves(out, main_items, "S1_", "S0_", lo, hi)
    cross = out[required].copy()
    out = _fill_within_wave(out, pd_items, "S1_", minimum["psychological_distance"], method, lo, hi)
    keep = out[required].notna().all(axis=1)

    audit = pd.DataFrame({"respondent_id": out["respondent_id"], "retained": keep})
    audit["n_filled_within_main"] = (original.isna() & within.notna()).sum(axis=1)
    audit["n_filled_cross_wave"] = (within.isna() & cross.notna()).sum(axis=1)
    audit["n_filled_pd"] = (cross.isna() & out[required].notna()).sum(axis=1)
    for name, items in groups.items():
        cols = [prefix + item for prefix in ["S0_", "S1_"] for item in items]
        audit[f"missing_{name}"] = out[cols].isna().any(axis=1)
    audit["missing_psychological_distance"] = out[["S1_" + x for x in pd_items]].isna().any(axis=1)
    # Full metadata table remains intact; only rows failing valid_cols are removed.
    out = out.loc[keep].copy()
    out = add_acceptance_consistency(out, acceptance_config["max_inconsistent_tasks"])
    if "S1_climate_blatten" in out:
        out["blatten_cc"] = out["S1_climate_blatten"].astype("string").str.contains("climate change", case=False, na=False)
    out = out.reset_index(drop=True)
    return (out, audit.reset_index(drop=True)) if return_audit else out


def validate_choice_tasks(df, left_option=1, right_option=2):
    if df.empty or left_option == right_option:
        raise ValueError("Choice data are empty or option codes are identical.")
    required = ["respondent_id", "task_id", "nh_event", "option", "chosen"]
    if df[required].isna().any().any():
        raise ValueError("Missing ID, wave, option or choice in conjoint data.")
    groups = df.groupby("task_id", sort=False)
    if not groups.size().eq(2).all() or not groups["option"].nunique().eq(2).all():
        raise ValueError("Each task must have exactly two distinct alternatives.")
    if not df["option"].isin([left_option, right_option]).all():
        raise ValueError("Unexpected alternative code.")
    if not df["chosen"].isin([0, 1]).all() or not groups["chosen"].sum().eq(1).all():
        raise ValueError("Each task needs exactly one chosen alternative (0/1).")
    if not groups[["respondent_id", "nh_event"]].nunique().eq(1).all().all():
        raise ValueError("Both alternatives must belong to the same person and wave.")
    if not df["nh_event"].isin([0, 1]).all():
        raise ValueError("The current model expects S0=0 and S1=1.")


def build_conjoint_long(df, config):
    """Create conjoint dataframe"""
    if df["respondent_id"].isna().any() or df["respondent_id"].duplicated().any():
        raise ValueError("One unique non-missing respondent_id per input row is required.")
    attribute_columns = df.filter(regex=r"^S\d+_choice\d+_(costs|benefits|exemptions)\d+$").columns.tolist()
    choice_columns = df.filter(regex=r"^S\d+_\d+_conjoint_prefer$").columns.tolist()
    if not attribute_columns or not choice_columns:
        raise ValueError("No conjoint attributes or choices found.")
    attributes = df.melt(id_vars="respondent_id", value_vars=attribute_columns, var_name="var", value_name="value")
    attributes[["nh_event", "task", "attribute", "option"]] = attributes["var"].str.extract(r"S(\d+)_choice(\d+)_(costs|benefits|exemptions)(\d+)")
    attributes[["nh_event", "task", "option"]] = attributes[["nh_event", "task", "option"]].astype(int)
    key = ["respondent_id", "nh_event", "task"]
    options = attributes.pivot(index=key + ["option"], columns="attribute", values="value").reset_index()
    options.columns.name = None
    choices = df.melt(id_vars="respondent_id", value_vars=choice_columns, var_name="var", value_name="choice")
    choices[["nh_event", "task"]] = choices["var"].str.extract(r"S(\d+)_(\d+)_conjoint_prefer").astype(int)
    choices["choice"] = pd.to_numeric(choices["choice"].map(lambda value: mp.PREFERENCE_MAP.get(value, value)), errors="coerce")
    out = options.merge(choices[key + ["choice"]], on=key, how="left", validate="many_to_one")
    out["chosen"] = out["choice"].eq(out["option"]).astype("int8")
    out = out.drop(columns="choice").sort_values(key + ["option"]).reset_index(drop=True)
    # Stable logical task key within the data version (rather than a positional index).
    out["task_id"] = out[key].astype(str).agg("|".join, axis=1)
    metadata = list(dict.fromkeys(["respondent_id"] + [c for c in config["respondent_columns"] if c in df]))
    out = out.merge(df[metadata], on="respondent_id", how="left", validate="many_to_one")
    validate_choice_tasks(out, config.get("left_option", 1), config.get("right_option", 2))
    return out


def design_columns(conjoint_config):
    return [f"{attribute}_{level}" for attribute, spec in conjoint_config["attributes"].items()
            for level in spec["levels"] if level != spec["baseline"]]


def encode_conjoint(df, config):
    """Encode categorical conjoint attributes for model estimation.

    Attribute levels and reference categories are defined in the conjoint
    configuration.
    - dummy coding represents the reference level by zeros,
    - effect coding represents it by -1 across the non-reference columns so that
    attribute-level coefficients sum to zero.

    Values that are missing or not listed as valid levels in the configuration
    raise an error instead of being treated as the reference category.
    """
    coding = config["coding"]
    if coding not in {"effect", "dummy"}:
        raise ValueError("conjoint.coding must be effect or dummy.")
    out = df.copy()
    for attribute, spec in config["attributes"].items():
        levels, baseline = spec["levels"], spec["baseline"]
        if len(levels) < 2 or len(set(levels)) != len(levels) or baseline not in levels:
            raise ValueError(f"Invalid level specification for {attribute}.")
        if not out[attribute].isin(levels).all():
            raise ValueError(f"Unknown or missing {attribute} levels: {out.loc[~out[attribute].isin(levels), attribute].unique()}")
        for level in levels:
            if level == baseline:
                continue
            values = out[attribute].eq(level).astype("int8")
            if coding == "effect":
                values = values - out[attribute].eq(baseline).astype("int8")
            out[f"{attribute}_{level}"] = values
    return out
