"""Reusable geographic assignment and cantonal map logic.

Geographic analysis of data.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .posterior import (
    align_respondents,
    check_model_coding,
    get_level_da,
    group_summary,
    quantity_da,
)
from .plotting import COUNT_CMAP, LEVEL_LABELS, LEVEL_ORDER, MAP_CMAP

def integer_codes(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values) & values.eq(np.floor(values))).astype("Int64")


def build_canton_lookup(localities, municipalities, cfg):
    """Retain all PLZ/BFS/canton combinations; never pick a canton arbitrarily."""
    lc = cfg["locality_columns"]
    mc = cfg["municipality_columns"]
    units = pd.DataFrame(
        {"municipality": integer_codes(municipalities[mc["municipality"]]),
         "canton": integer_codes(municipalities[mc["canton"]]),}
    ).dropna().drop_duplicates()
    if units.municipality.duplicated().any():
        raise ValueError("A BFS municipality maps to several cantons; check geodata versions/layers.")

    locations = pd.DataFrame(
        {"postal_code": integer_codes(localities[lc["postal_code"]]),
         "municipality": integer_codes(localities[lc["municipality"]]),}
    )
    locality_col = lc.get("locality")
    if locality_col and locality_col in localities:
        locations["locality"] = localities[locality_col].astype("string").str.strip().str.casefold()
    else:
        locations["locality"] = pd.NA

    return locations.merge(units, on="municipality", how="left", validate="many_to_one").drop_duplicates()


def assign_cantons(metadata, lookup, cfg,):
    """Assign one canton per respondent when the evidence is unique/resolvable."""
    if metadata.respondent_id.isna().any() or metadata.respondent_id.duplicated().any():
        raise ValueError("Geocoding requires one unique respondent_id per row.")

    postal_col = cfg["respondent_postal_column"]
    if postal_col not in metadata:
        raise KeyError(f"Postal-code column missing: {postal_col}")
    postal = integer_codes(metadata[postal_col])

    rows = []
    for index, person in metadata.iterrows():
        code = postal.loc[index]
        candidates = lookup.loc[lookup.postal_code.eq(code).fillna(False)].dropna(subset=["canton"])
        cantons = candidates.canton.unique()
        assigned = pd.NA
        status = "missing_postcode" if pd.isna(code) else "unmatched_postcode"

        if len(cantons) == 1:
            assigned, status = int(cantons[0]), "unique_postcode"
        elif len(cantons) > 1:
            status = "ambiguous_postcode"
            selected = candidates
            if len(selected.canton.unique()) == 1:
                assigned, status = int(selected.canton.iloc[0]), "resolved_with_locality"

        rows.append(
            {
                "respondent_id": person.respondent_id,
                "respondent_key": person.respondent_key,
                "postal_code": code,
                "canton": assigned,
                "geo_status": status,
                "candidate_cantons": ",".join(
                    map(str, sorted(map(int, cantons)))
                ),
                "override_reason": "",
            }
        )

    result = pd.DataFrame(rows)
    result["canton"] = result.canton.astype("Int64")
    return result


def merge_geography(metadata, geography):
    merged = metadata.merge(
        geography[["respondent_id", "respondent_key", "canton", "geo_status"]],
        on="respondent_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_geo"),
    )
    if not merged.respondent_key.eq(merged.respondent_key_geo).all():
        raise ValueError("Geography and survey tables belong to different respondent mappings.")
    return merged.drop(columns="respondent_key_geo")


def survey_summary(metadata, geography, spec):
    """Aggregate one configured survey quantity by canton."""
    frame = merge_geography(metadata, geography)
    if spec["type"] == "count":
        table = (
            frame.dropna(subset=["canton"])
            .groupby("canton")
            .size()
            .rename("mean")
            .reset_index()
        )
        table["n_respondents"] = table["mean"]
        table["n_valid"] = table["mean"]
        return table

    values = frame[spec["columns"]].apply(pd.to_numeric, errors="coerce")
    if "valid_range" in spec:
        lo, hi = spec["valid_range"]
        values = values.where((values >= lo) & (values <= hi))

    operation = spec.get("operation", "mean")
    if operation == "mean":
        score = values.mean(axis=1).where(values.notna().sum(axis=1).ge(spec.get("min_items", 1)))
    elif operation == "difference" and values.shape[1] == 2:
        score = values.iloc[:, 1] - values.iloc[:, 0]
    elif operation == "any":
        score = values.eq(1).any(axis=1).astype(float)
        score = score.where(values.notna().all(axis=1) | score.eq(1))
    else:
        raise ValueError("Survey map operation must be mean, any, or two-column difference.")

    frame["score"] = score
    return (
        frame.dropna(subset=["canton"])
        .groupby("canton")
        .agg(
            mean=("score", "mean"),
            n_respondents=("respondent_id", "size"),
            n_valid=("score", "count"),
        )
        .reset_index()
    )


def posterior_map_summaries(idata, metadata, geography, cfg, spec, hdi_prob):
    """Aggregate posterior quantities per draw and then summarize by canton."""
    check_model_coding(idata, cfg)
    aligned = merge_geography(align_respondents(idata, metadata), geography)

    if spec["type"] == "trait":
        da = idata.posterior[spec["variable"]]
        table = group_summary(da, aligned, "canton", hdi_prob)
        table["n_valid"] = table.n_respondents
        return [(spec.get("label", spec["variable"]), table)]

    if spec["type"] != "utility":
        raise ValueError("Posterior map type must be trait or utility.")

    attr = spec["attribute"]
    da = quantity_da(idata, spec.get("quantity", "shift"), individual=True)
    output = []
    for level in spec.get("levels", LEVEL_ORDER[attr]):
        table = group_summary(get_level_da(da, attr, level, cfg), aligned, "canton", hdi_prob)
        table["n_valid"] = table.n_respondents
        table["attribute"] = attr
        table["level"] = level
        output.append((LEVEL_LABELS.get(level, level), table))
    return output


def prepare_canton_geometry(source, cfg):
    """Prepare canton polygons from an already loaded GeoDataFrame."""
    columns = cfg["canton_columns"]
    if source.crs is None:
        raise ValueError("Boundary file has no CRS; do not guess the projection.")
    source = source.copy()
    source["canton"] = integer_codes(source[columns["canton"]])
    source["canton_name"] = source[columns["name"]]
    return (
        source[["canton", "canton_name", "geometry"]]
        .dropna(subset=["canton"])
        .dissolve(by="canton", as_index=False)
    )


def draw_canton_maps(geometry, summaries, spec, cfg):
    """Plots geographic heatmaps (cantonwise) to depict model and survey results.
    
    Draw borders/hatching/Blatten marker with one scale across panels."""
    import geopandas as gpd
    from shapely.geometry import Point

    count = len(summaries)
    if count == 0:
        raise ValueError("No map panels requested.")

    ncols = min(int(spec.get("ncols", 3)), count)
    nrows = int(np.ceil(count / ncols))
    single = count == 1
    size = tuple(spec.get("figsize", (9, 12) if single else (6 * ncols + 0.8, 5 * nrows)))

    if single:
        fig, single_ax = plt.subplots(figsize=size)
    else:
        fig = plt.figure(figsize=size)
        grids = fig.add_gridspec(
            nrows,
            ncols + 1,
            width_ratios=[1] * ncols + [0.05],
            wspace=0.08,
            hspace=0.16,
            left=0.03,
            right=0.96,
            top=0.87,
            bottom=0.13,
        )

    arrays = [table["mean"].dropna().to_numpy(float) for _, table in summaries]
    values = np.concatenate(arrays) if arrays else np.array([])
    if not values.size:
        raise ValueError("No mapped values: inspect the geographic assignment audit.")

    # Colormap is a presentation choice, not a scientific config option.
    cmap = COUNT_CMAP if spec["type"] == "count" else MAP_CMAP
    if spec.get("center_zero", False):
        vmax = spec.get("vmax", max(float(np.abs(values).max()), 0.01))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        vmin = spec.get("vmin", float(values.min()))
        vmax = spec.get("vmax", float(values.max()))
        if vmin == vmax:
            vmin -= 0.01
            vmax += 0.01
        center = spec.get("vcenter")
        norm = (
            TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            if center is not None
            else Normalize(vmin=vmin, vmax=vmax)
        )

    threshold = int(cfg.get("small_sample_threshold", 20))
    map_axes = []
    for i, (label, table) in enumerate(summaries):
        ax = single_ax if single else fig.add_subplot(grids[i // ncols, i % ncols])
        map_axes.append(ax)
        mapped = geometry.merge(table, on="canton", how="left", validate="one_to_one")
        mapped.plot(
            column="mean",
            ax=ax,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.6,
            legend=False,
            missing_kwds={"color": "lightgrey", "edgecolor": "black"},
        )
        low = mapped["n_valid"].lt(threshold) & mapped["mean"].notna()
        if low.any():
            mapped.loc[low].plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=0.65,
                hatch="///",
            )

        # Blatten marker
        marker = cfg.get("event_marker")
        if marker:
            point = (
                gpd.GeoSeries([Point(marker["longitude"], marker["latitude"])], crs=4326)
                .to_crs(mapped.crs)
                .iloc[0]
            )
            ax.scatter(
                point.x,
                point.y,
                s=100,
                marker="*",
                color="black",
                edgecolor="white",
                linewidth=0.8,
                zorder=10,
            )
        if spec.get("title"):
            ax.set_title(textwrap.fill(label, 48), fontsize=15, pad=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if single:
        cbar = fig.colorbar(sm, ax=single_ax, fraction=0.03, pad=0.015, shrink=0.37)
    else:
        cbar = fig.colorbar(sm, cax=fig.add_subplot(grids[:, -1]))
    cbar.set_label(spec.get("colorbar_label", spec.get("label", "Mean")))

    handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label=f"n < {threshold}",
        ),
        Patch(facecolor="lightgrey", edgecolor="black", label="No mapped data"),
    ]
    if single:
        single_ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=12)
    else:
        fig.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.03, 0.025),
            frameon=True,
            fontsize=12,
        )
    if spec.get("title"):
        fig.suptitle(spec["title"], y=0.96, fontsize=17, fontweight="bold")
    for ax in map_axes:
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.tick_params(left=False, right=False, top=False, bottom=False,
                    labelleft=False, labelright=False, labeltop=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig
