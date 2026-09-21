"""Descriptive Likert figures. Scientific plotting logic lives in src/."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from natural_hazard_solidarity.descriptive_plots import (
    plot_deservingness_likert,
    plot_likert_shifts,
    plot_pre_post_likert,
)
from natural_hazard_solidarity.plotting import save_figure


def main(s):
    frame = pd.read_parquet(s.input.analysis)
    kind = str(s.params.kind)
    if kind == "pre_post":
        fig = plot_pre_post_likert(frame, dict(s.params.figure))
    elif kind == "deservingness":
        fig = plot_deservingness_likert(frame)
    elif kind == "likert_shifts":
        fig = plot_likert_shifts(frame)
    else:
        raise ValueError(f"Unknown descriptive plot kind: {kind}")
    save_figure(fig, s.output.figure, dpi=800)


if __name__ == "__main__":
    main(globals()["snakemake"])
