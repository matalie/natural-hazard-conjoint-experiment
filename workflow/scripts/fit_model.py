"""Run one model; safe to import in Windows multiprocessing child processes."""
from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone
import importlib.metadata as metadata
import pandas as pd
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from natural_hazard_solidarity.data_preparation import filter_model_sample
from natural_hazard_solidarity.models import (prepare_model_data, build_hcm, build_simple_choice_model,
                                             build_longitudinal_mixed_logit, choice_signature)


def main(snakemake):
    import pymc as pm
    run, sampling = dict(snakemake.params.run), dict(snakemake.params.sampling)
    family = run.get("family", "hcm")
    frame = filter_model_sample(pd.read_parquet(snakemake.input.conjoint), snakemake.params.sample)
    if frame.empty:
        raise ValueError(f"Run {snakemake.params.run_name}: empty sample.")
    data = prepare_model_data(frame, snakemake.params.conjoint, snakemake.params.constructs,
                              include_constructs=(family == "hcm"))
    if family == "hcm":
        model = build_hcm(data, run["prior_factor"], run.get("include_financial_vulnerability", True))
    elif family == "simple":
        model = build_simple_choice_model(data)
    elif family == "mixed_logit":
        model = build_longitudinal_mixed_logit(data)
    else:
        raise ValueError(f"Unknown model family: {family}; adjusted is deliberately not supported.")
    cores = min(snakemake.threads, sampling["chains"])
    blas_threads = int(sampling.get("blas_threads_per_chain", 1))
    if blas_threads != 1:
        raise ValueError("This workflow budgets one BLAS thread per chain; keep blas_threads_per_chain=1.")
    with model:
        idata = pm.sample(draws=sampling["draws"], tune=sampling["tune"], chains=sampling["chains"],
                          cores=cores, blas_cores=cores * blas_threads,
                          random_seed=sampling["random_seed"], target_accept=sampling["target_accept"],
                          progressbar=sampling.get("progressbar", True), return_inferencedata=True,
                          idata_kwargs={"log_likelihood": sampling.get("log_likelihood", True)})
    idata.add_groups({"run_data": xr.Dataset(
        {"respondent_key": ("respondent", data["respondent_keys"])},
        coords={"respondent": data["respondent_ids"]})})
    idata.posterior.attrs.update({
        "run_name": str(snakemake.params.run_name), "family": family, "sample": run["sample"],
        "coding": snakemake.params.conjoint["coding"], "id_scheme": "respondent_id_v2",
        "n_respondents": data["n_respondents"], "n_tasks": len(data["task_ids"]),
        "choice_signature": choice_signature(data), "run_json": json.dumps(run, sort_keys=True),
        "sampling_json": json.dumps(sampling, sort_keys=True),
        "standardization_json": json.dumps(data["standardization"], sort_keys=True),
        "conjoint_config_json": json.dumps(snakemake.params.conjoint, sort_keys=True),
        "conjoint_sha256": hashlib.sha256(Path(snakemake.input.conjoint).read_bytes()).hexdigest(),
        "versions_json": json.dumps({p: metadata.version(p) for p in ["pymc", "pytensor", "arviz", "numpy", "pandas"]}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    output = Path(snakemake.output.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(output), engine="h5netcdf")
    print(f"Saved {output}: {data['n_respondents']} respondents, {len(data['task_ids'])} tasks")


if __name__ == "__main__":
    main(globals()["snakemake"])
