"""Snakemake entry point for one H1-H4 hypothesis figure."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import arviz as az

from natural_hazard_solidarity.hypothesis_plots import plot_hypothesis
from natural_hazard_solidarity.plotting import save_figure


def main(s) -> None:
    idata = az.from_netcdf(s.input.model)

    figure_config = dict(s.params.figure)
    figure_config.setdefault("hdi_prob", float(s.params.hdi_prob))
    conjoint_config = s.params.conjoint

    hypothesis = figure_config["hypothesis"]

    fig, table = plot_hypothesis(
        idata=idata,
        hypothesis=hypothesis,
        conjoint_config=conjoint_config,
        plot_config=figure_config,
    )

    save_figure(fig, s.output.figure)

    table_path = Path(s.output.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    print(f"Saved {hypothesis}: {s.output.figure}")
    print(f"Saved table: {s.output.table}")


if __name__ == "__main__":
    main(globals()["snakemake"])