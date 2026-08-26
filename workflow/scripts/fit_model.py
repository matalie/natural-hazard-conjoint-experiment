from pathlib import Path
import sys
import pandas as pd
import pymc as pm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from natural_hazard_solidarity.data_preparation import filter_model_sample  # noqa: E402
from natural_hazard_solidarity.models import build_hcm, prepare_model_data  # noqa: E402


def main(snakemake) -> None:
    run_name = snakemake.params.run_name
    run = snakemake.params.run
    sampling = snakemake.params.sampling

    conjoint = pd.read_parquet(snakemake.input.conjoint)
    sample = snakemake.params.samples[run["sample"]]
    model_frame = filter_model_sample(conjoint, sample_config=sample)

    if model_frame.empty:
        raise ValueError(f"Model run '{run_name}' produced an empty sample.")

    data = prepare_model_data(
        model_frame,
        conjoint_config=snakemake.params.conjoint,
        constructs=snakemake.params.constructs,
    )

    model = build_hcm(
        data,
        prior_factor=run["prior_factor"],
        include_financial_vulnerability=run.get("include_financial_vulnerability", True),
    )

    with model:
        idata = pm.sample(
            draws=sampling["draws"],
            tune=sampling["tune"],
            chains=sampling["chains"],
            cores=snakemake.threads,
            random_seed=sampling["random_seed"],
            target_accept=sampling["target_accept"],
            progressbar=sampling.get("progressbar", True),
            idata_kwargs={"log_likelihood": sampling.get("log_likelihood", True)},
        )

    idata.posterior.attrs["run_name"] = run_name
    idata.posterior.attrs["sample"] = run["sample"]
    idata.posterior.attrs["coding"] = snakemake.params.conjoint["coding"]
    idata.posterior.attrs["prior_factor"] = float(run["prior_factor"])

    output = Path(snakemake.output.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(output)

    print(
        f"Saved {run_name}: {output} "
        f"({model_frame['respondent_id'].nunique()} respondents, "
        f"{model_frame['task_id'].nunique()} tasks)"
    )


if __name__ == "__main__":
    main(globals()["snakemake"])