"""Thin Snakemake entry point for respondent-level HCM figures."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from natural_hazard_solidarity.posterior import level_shift_decomposition
from natural_hazard_solidarity.respondent_plots import (
    plot_latent_constructs,
    plot_level_decomposition_by_respondent,
)
from natural_hazard_solidarity.plotting import save_figure


def main(s):
    import arviz as az

    idata = az.from_netcdf(s.input.model)
    conjoint = dict(s.params.conjoint)

    latent_fig = plot_latent_constructs(idata)
    save_figure(latent_fig, s.output.latent, dpi=600)

    specs = list(s.params.specifications)
    figures = list(s.output.impacts)
    tables = list(s.output.tables)
    if not (len(specs) == len(figures) == len(tables)):
        raise ValueError("Respondent impact specifications, figures and tables must align.")

    pretty_attr = {"costs": "Costs", "exemptions": "Exemptions", "benefits": "Benefits"}
    for spec, figure, table in zip(specs, figures, tables):
        attribute, level = spec["attribute"], spec["level"]
        decomposition = level_shift_decomposition(
            idata,
            attribute,
            level,
            conjoint,
            hdi_prob=float(s.params.hdi_prob),
        )
        fig = plot_level_decomposition_by_respondent(
            decomposition,
            title=f"{pretty_attr.get(attribute, attribute)}: {level}",
        )
        save_figure(fig, figure, dpi=800)

        output = Path(table)
        output.parent.mkdir(parents=True, exist_ok=True)
        decomposition.to_csv(output, index=False)


if __name__ == "__main__":
    main(globals()["snakemake"])
