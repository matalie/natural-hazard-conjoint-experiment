"""Snakemake entry point for generating the posterior 
    parameter tables of the main model."""

from pathlib import Path
import arviz as az
from natural_hazard_solidarity.diagnostic_tables import (
    latent_loading_summary,
    main_model_summary,
)

def main(s):
    idata = az.from_netcdf(s.input.model)

    main_table = main_model_summary(idata)
    latent_table = latent_loading_summary(idata)

    Path(s.output.main).parent.mkdir(parents=True, exist_ok=True)
    Path(s.output.latent).parent.mkdir(parents=True, exist_ok=True)

    main_table.to_csv(s.output.main, index=False, float_format="%.3f",)
    latent_table.to_csv(s.output.latent, index=False, float_format="%.3f",)

    print(f"Saved table: {s.output.main}")
    print(f"Saved table: {s.output.latent}")


if __name__ == "__main__":
    main(globals()["snakemake"])