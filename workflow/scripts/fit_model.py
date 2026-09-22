"""Snakemake adapter for fitting and saving one configured model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json

import pandas as pd
import xarray as xr

# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from natural_hazard_solidarity.data_preparation import filter_model_sample
from natural_hazard_solidarity.models import choice_signature, fit_model


def main(s):
    run = dict(s.params.run)
    sampling = {**dict(s.params.sampling), **run.get("sampling", {})}

    frame = pd.read_parquet(s.input.conjoint)
    frame = filter_model_sample(frame, s.params.sample)

    if frame.empty:
        raise ValueError(f"Run {s.params.run_name}: empty sample.")

    idata, data = fit_model(
        frame=frame,
        run=run,
        sampling=sampling,
        conjoint_config=s.params.conjoint,
        constructs=s.params.constructs,
        max_cores=s.threads,
    )

    idata.add_groups({
        "run_data": xr.Dataset(
            {"respondent_key": ("respondent", data["respondent_keys"])},
            coords={"respondent": data["respondent_ids"]},
        )
    })

    idata.posterior.attrs.update({
        "run_name": str(s.params.run_name),
        "family": run.get("family", "hcm"),
        "sample": run["sample"],
        "coding": s.params.conjoint["coding"],
        "id_scheme": "respondent_id_v2",
        "n_respondents": data["n_respondents"],
        "n_tasks": len(data["task_ids"]),
        "choice_signature": choice_signature(data),
        "run_json": json.dumps(run, sort_keys=True),
        "sampling_json": json.dumps(sampling, sort_keys=True),
        "standardization_json": json.dumps(data["standardization"], sort_keys=True),
        "conjoint_config_json": json.dumps(s.params.conjoint, sort_keys=True),
        "conjoint_sha256": hashlib.sha256(Path(s.input.conjoint).read_bytes()).hexdigest(),
        "versions_json": json.dumps({
            package: metadata.version(package)
            for package in ["pymc", "pytensor", "arviz", "numpy", "pandas"]
        }),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })

    output = Path(s.output.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(output, engine="h5netcdf")

    print(
        f"Saved {output}: {data['n_respondents']} respondents, "
        f"{len(data['task_ids'])} tasks"
    )


if __name__ == "__main__":
    main(globals()["snakemake"])
