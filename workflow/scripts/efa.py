"""Snakemake entry point for EFA and construct-correlation figures of the latent construct items."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from natural_hazard_solidarity.efa import (
    complete_efa_items,
    fit_efa,
    plot_correlation_matrix,
    plot_efa_comparison,
)
from natural_hazard_solidarity.plotting import save_figure


def main(s):
    frame = pd.read_parquet(s.input.analysis)
    attributes = complete_efa_items(frame)
    correlation_fig, correlation = plot_correlation_matrix(attributes)
    save_figure(correlation_fig, s.output.correlation, dpi=600)
    results, diagnostics = fit_efa(attributes)
    efa_fig = plot_efa_comparison(results)
    save_figure(efa_fig, s.output.comparison, dpi=600)

    print(
        "EFA:",
        f"N_complete={diagnostics['n_complete']}",
        f"KMO={diagnostics['kmo_model']:.3f}",
        f"Bartlett chi2={diagnostics['bartlett_chi2']:.3f}",
        f"p={diagnostics['bartlett_p']:.6g}",
    )
    # Useful numerical companions are deliberately kept as rule outputs.
    Path(s.output.correlation_table).parent.mkdir(parents=True, exist_ok=True)
    correlation.to_csv(s.output.correlation_table)
    results[2]["loadings"].to_csv(s.output.loadings_2)
    results[3]["loadings"].to_csv(s.output.loadings_3)
    results[2]["phi_df"].to_csv(s.output.phi_2)
    results[3]["phi_df"].to_csv(s.output.phi_3)


if __name__ == "__main__":
    main(globals()["snakemake"])
