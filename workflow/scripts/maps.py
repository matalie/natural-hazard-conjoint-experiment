"""Thin Snakemake entry point for geographic assignment and canton maps."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

# import sys
# SNAKEMAKE = globals().get("snakemake")
# if SNAKEMAKE is not None:
#     REPO_ROOT = Path(SNAKEMAKE.scriptdir).parents[1]
# else:
#     REPO_ROOT = Path.cwd()

# sys.path.insert(0, str(REPO_ROOT / "src"))

from natural_hazard_solidarity.maps import (
    assign_cantons,
    build_canton_lookup,
    draw_canton_maps,
    posterior_map_summaries,
    prepare_canton_geometry,
    survey_summary,
)
from natural_hazard_solidarity.plotting import save_figure


def run_assignment(s):
    import geopandas as gpd

    cfg = dict(s.params.geography)
    metadata = pd.read_parquet(s.input.respondents)
    localities = pd.read_csv(
        s.input.localities, sep=cfg.get("locality_sep", ";")
    )
    municipalities = gpd.read_file(
        s.input.boundaries, layer=cfg["municipality_layer"]
    )
    lookup = build_canton_lookup(localities, municipalities, cfg)


    assignment = assign_cantons(metadata, lookup, cfg)
    assignment_path = Path(s.output.assignments)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_parquet(assignment_path, index=False)

    audit_path = Path(s.output.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(audit_path, index=False)
    print(assignment.geo_status.value_counts().to_string())


def run_map(s):
    import geopandas as gpd

    cfg = dict(s.params.geography)
    spec = dict(s.params.figure)
    metadata = pd.read_parquet(s.input.respondents)
    geography = pd.read_parquet(s.input.assignments)

    if s.params.stage == "survey":
        summaries = [(spec["label"], survey_summary(metadata, geography, spec))]
    elif s.params.stage == "posterior":
        import arviz as az

        idata = az.from_netcdf(s.input.model)
        summaries = posterior_map_summaries(
            idata=idata,
            metadata=metadata,
            geography=geography,
            cfg=dict(s.params.conjoint),
            spec=spec,
            hdi_prob=float(s.params.hdi_prob),
        )
    else:
        raise ValueError(f"Unknown map stage: {s.params.stage}")

    source = gpd.read_file(s.input.boundaries, layer=cfg["canton_layer"])
    geometry = prepare_canton_geometry(source, cfg)
    fig = draw_canton_maps(geometry, summaries, spec, cfg)
    save_figure(fig, s.output.figure)

    table = pd.concat(
        [table.assign(panel=label) for label, table in summaries],
        ignore_index=True,
    )
    table_path = Path(s.output.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)


def main(s):
    if s.params.stage == "assign":
        run_assignment(s)
    else:
        run_map(s)


if __name__ == "__main__":
    main(globals()["snakemake"])
