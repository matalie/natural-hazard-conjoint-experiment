"""Thin Snakemake entry point for posterior diagnostic figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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
