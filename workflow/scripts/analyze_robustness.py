"""Thin Snakemake entry point for one robustness comparison."""
from pathlib import Path
# import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

# SNAKEMAKE = globals().get("snakemake")
# if SNAKEMAKE is not None:
#     REPO_ROOT = Path(SNAKEMAKE.scriptdir).parents[1]
# else:
#     REPO_ROOT = Path.cwd()

# sys.path.insert(0, str(REPO_ROOT / "src"))

from natural_hazard_solidarity.plotting import save_figure
from natural_hazard_solidarity.robustness_analysis import (
    assert_sample_audit,
    build_sample_audit,
    plot_robustness,
)


def main(s):
    import arviz as az

    spec = dict(s.params.specification)
    run_names = list(spec["runs"])
    models = {
        name: az.from_netcdf(path)
        for name, path in zip(run_names, s.input.models)
    }

    conjoint_frame = pd.read_parquet(s.input.conjoint)
    audit = build_sample_audit(
        models=models,
        specification=spec,
        run_configs=dict(s.params.runs),
        sample_configs=dict(s.params.samples),
        conjoint_frame=conjoint_frame,
    )
    audit_path = Path(s.output.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print("Robustness sample audit:")
    print(audit.to_string(index=False))
    assert_sample_audit(audit)

    fig, table = plot_robustness(
        models=models,
        specification=spec,
        conjoint_config=dict(s.params.conjoint),
        hdi_prob=float(s.params.hdi_prob),
    )
    save_figure(fig, s.output.figure)

    table_path = Path(s.output.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)


if __name__ == "__main__":
    main(globals()["snakemake"])
