"""Thin Snakemake entry point for one H1-H4 hypothesis figure."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import arviz as az


SNAKEMAKE = globals().get("snakemake")
if SNAKEMAKE is not None:
    REPO_ROOT = Path(SNAKEMAKE.scriptdir).parents[1]
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT / "src"))

from natural_hazard_solidarity.hypothesis_plots import plot_hypothesis
from natural_hazard_solidarity.plotting import save_figure


def main(snakemake) -> None:
    idata = az.from_netcdf(snakemake.input.model)

    figure_config = snakemake.params.figure
    conjoint_config = snakemake.params.conjoint

    hypothesis = figure_config["hypothesis"]

    fig, table = plot_hypothesis(
        idata=idata,
        hypothesis=hypothesis,
        conjoint_config=conjoint_config,
        plot_config=figure_config,
    )

    save_figure(fig, snakemake.output.figure)

    table_path = Path(snakemake.output.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    print(f"Saved {hypothesis}: {snakemake.output.figure}")
    print(f"Saved table: {snakemake.output.table}")


if __name__ == "__main__":
    main(globals()["snakemake"])