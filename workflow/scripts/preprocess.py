"""Three coarse preprocessing stages; no notebooks, no sampling."""
from pathlib import Path
import json
import hashlib
import pandas as pd

# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from natural_hazard_solidarity.data_preparation import (clean_wave, merge_waves, prepare_analysis_sample,
                                                      build_conjoint_long, encode_conjoint)


def save(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main(s):
    stage = s.params.stage
    if stage == "clean":
        waves = {}
        for wave, path in [("S0", s.input.s0), ("S1", s.input.s1)]:
            raw = pd.read_csv(path, skiprows=s.params.preprocessing["skiprows"], dtype={"id": "string"},
                              sep=s.params.preprocessing.get("csv_sep", ","))
            waves[wave] = clean_wave(raw, wave, s.params.preprocessing)
        ids = pd.read_csv(s.input.ids, sep=s.params.id_mapping["sep"])
        combined = merge_waves(waves["S0"], waves["S1"], ids, s.params.id_mapping)
        save(waves["S0"], s.output.s0); save(waves["S1"], s.output.s1); save(combined, s.output.combined)
        report = {"S0_clean": len(waves["S0"]), "S1_clean": len(waves["S1"]), "linked": len(combined),
                  "raw_sha256": {wave: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for wave, path in [("S0", s.input.s0), ("S1", s.input.s1)]}}
        Path(s.output.report).parent.mkdir(parents=True, exist_ok=True)
        Path(s.output.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report)
    elif stage == "sample":
        combined = pd.read_parquet(s.input.combined)
        analysis, audit = prepare_analysis_sample(
            combined,
            s.params.analysis_items,
            s.params.imputation,
            s.params.acceptance,
            return_audit=True,
        )
        save(analysis, s.output.analysis)
        Path(s.output.audit).parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(s.output.audit, index=False)
        print(f"Linked={len(combined)}, retained={len(analysis)}, excluded={len(combined)-len(analysis)}")
    elif stage == "conjoint":
        analysis = pd.read_parquet(s.input.analysis)
        result = encode_conjoint(build_conjoint_long(analysis, s.params.conjoint), s.params.conjoint)
        save(result, s.output.conjoint)
        print(f"Conjoint: {result.respondent_id.nunique()} respondents, {result.task_id.nunique()} tasks, {len(result)} alternatives")
    else:
        raise ValueError(f"Unknown preprocessing stage: {stage}")


if __name__ == "__main__":
    main(globals()["snakemake"])
