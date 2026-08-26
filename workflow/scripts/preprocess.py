from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

# Snakemake executes scripts from a temporary location. scriptdir points to the
# original workflow/scripts directory, so this makes src/ importable without
# requiring an editable package installation inside every Conda environment.
REPO_ROOT = Path(snakemake.scriptdir).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from natural_hazard_solidarity.data_preparation import (  # noqa: E402
    build_conjoint_long,
    clean_wave,
    encode_conjoint,
    merge_waves,
    prepare_analysis_sample,
)

def save_parquet(frame: pd.DataFrame, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)

raw_s0 = pd.read_csv(
    snakemake.input.s0,
    dtype={"id": "string"},
    skiprows=snakemake.params.skiprows,
)
raw_s1 = pd.read_csv(
    snakemake.input.s1,
    dtype={"id": "string"},
    skiprows=snakemake.params.skiprows,
)

s0 = clean_wave(raw_s0, wave="S0", config=snakemake.params.preprocessing)
s1 = clean_wave(raw_s1, wave="S1", config=snakemake.params.preprocessing)
save_parquet(s0, snakemake.output.s0)
save_parquet(s1, snakemake.output.s1)

keys = pd.read_csv(
    snakemake.input.ids,
    sep=snakemake.params.id_mapping["sep"],
)
combined = merge_waves(s0, s1, keys, config=snakemake.params.id_mapping)
save_parquet(combined, snakemake.output.combined)

analysis_sample = prepare_analysis_sample(
    combined,
    item_config=snakemake.params.analysis_items,
    imputation_config=snakemake.params.preprocessing["imputation"],
    acceptance_config=snakemake.params.acceptance,
)
save_parquet(analysis_sample, snakemake.output.analysis)

conjoint = build_conjoint_long(analysis_sample, config=snakemake.params.conjoint)
conjoint = encode_conjoint(conjoint, config=snakemake.params.conjoint)
save_parquet(conjoint, snakemake.output.conjoint)

print(
    "Preprocessing finished: "
    f"S0={len(s0)}, S1={len(s1)}, linked respondents={len(analysis_sample)}, "
    f"conjoint rows={len(conjoint)}"
)
