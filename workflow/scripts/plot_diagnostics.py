"""Snakemake entry point for posterior diagnostic figures.

    Includes plots of the posterior distribution per model chain.
    And posterior values per chain and tuning point.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from natural_hazard_solidarity.diagnostic_plots import plot_partworth_trace
from natural_hazard_solidarity.plotting import save_figure

def main(s):
    import arviz as az

    idata = az.from_netcdf(s.input.model)
    conjoint = dict(s.params.conjoint)
    variable = str(getattr(s.params, "variable", "partworth_mean"))
    fig = plot_partworth_trace(idata, conjoint, variable=variable)
    save_figure(fig, s.output.figure, dpi=600)

if __name__ == "__main__":
    main(globals()["snakemake"])
