import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from natural_hazard_solidarity.plotting import (
    ATTRIBUTE_COLORS,
    posterior_interval,
    save_figure,
    set_thesis_style,
    plot_intervals,
)

set_thesis_style()


LEVEL_LABELS = {
    "All people pay the same amount": "Equal payments",
    "People pay proportionally to their income": "Income proportional",
    "People & companies being protected by protective measures": "Beneficiaries pay",
    "People pay proportionally to their CO2 emissions": "Individuals: CO₂ emissions",
    "Companies pay proportionally to their CO2 emissions": "Companies: CO₂ emissions",

    "No groups exempted from costs": "No exemptions",
    "Low-income earners exempted from costs": "Low income exempt",
    "Low- and middle-income earners exempted from costs": "Low & middle income exempt",

    "Equal protection levels for all municipalities": "Equal protection",
    "Municipalities most affected by natural hazards even if they are economically declining":
        "Most hazard-affected",
    "Economically prosperous municipalities": "Economically prosperous",
    "Municipalities in which people have lived in for many years should be protected at all costs":
        "Long-established communities",
    "Culturally valuable municipalities e.g. with historic buildings":
        "Culturally valuable",
}


def _short_label(level: str) -> str:
    return LEVEL_LABELS.get(level, level)


def _plot_h1_panel(ax, groups, draws, title):
    positions = []
    labels = []
    group_positions = {}
    y = 0

    for attribute, levels in groups:
        start = y

        for level in levels:
            mean, lower, upper = posterior_interval(draws[level])

            ax.errorbar(
                mean,
                y,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o",
                capsize=3,
                color=ATTRIBUTE_COLORS[attribute],
                ecolor=ATTRIBUTE_COLORS[attribute],
            )

            positions.append(y)
            labels.append(_short_label(level))
            y += 1

        group_positions[attribute] = (start, y - 1)
        y += 1

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Utility")
    ax.set_title(title)
    ax.invert_yaxis()

    for attribute, (start, end) in group_positions.items():
        ax.text(
            -0.04,
            start - 0.55,
            attribute.capitalize(),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontweight="bold",
            color=ATTRIBUTE_COLORS[attribute],
        )


def plot_h1(idata, output, conjoint_config):
    posterior = idata.posterior

    pre = posterior["partworth_pre_mean"]
    shift = posterior["shift_mean"]

    groups = []
    pre_draws = {}
    shift_draws = {}

    for attribute, specification in conjoint_config["attributes"].items():
        baseline = specification["baseline"]
        levels = specification["levels"]
        nonbaseline = [level for level in levels if level != baseline]

        pre_coefficients = []
        shift_coefficients = []

        for level in nonbaseline:
            feature = f"{attribute}_{level}"

            pre_level = pre.sel(level=feature)
            shift_level = shift.sel(level=feature)

            pre_coefficients.append(pre_level)
            shift_coefficients.append(shift_level)

            pre_draws[level] = pre_level.values
            shift_draws[level] = shift_level.values

        if conjoint_config["coding"] == "effect":
            pre_draws[baseline] = (-sum(pre_coefficients)).values
            shift_draws[baseline] = (-sum(shift_coefficients)).values
        else:
            pre_draws[baseline] = np.zeros_like(pre_coefficients[0].values)
            shift_draws[baseline] = np.zeros_like(shift_coefficients[0].values)

        groups.append((attribute, levels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    _plot_h1_panel(
        axes[0],
        groups,
        pre_draws,
        "Pre-event utility",
    )

    _plot_h1_panel(
        axes[1],
        groups,
        shift_draws,
        "Utility shift",
    )

    fig.tight_layout()
    save_figure(fig, output)


def plot_h2(idata, output):
    """H2: Association of psychological distance with conjoint preferences."""
    posterior = idata.posterior

    pre = posterior["gamma_pd"]
    shift = posterior["shift_psy"]
    post = pre + shift

    feature_dim = next(dim for dim in pre.dims if dim not in {"chain", "draw"})
    features = pre.coords[feature_dim].values

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, values, title in zip(
        axes,
        [pre, shift, post],
        [
            "Pre-event association",
            "Change associated with psychological distance",
            "Post-event association",
        ],
    ):
        draws = {
            str(feature): values.sel({feature_dim: feature}).values
            for feature in features
        }
        plot_intervals(ax, draws)
        ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output)


def plot_h3(idata, output):
    """H3: Association of financial vulnerability with conjoint preferences."""
    posterior = idata.posterior

    pre = posterior["gamma_fv"]
    shift = posterior["shift_fin"]
    post = pre + shift

    feature_dim = next(dim for dim in pre.dims if dim not in {"chain", "draw"})
    features = pre.coords[feature_dim].values

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, values, title in zip(
        axes,
        [pre, shift, post],
        [
            "Pre-event association",
            "Change associated with financial vulnerability",
            "Post-event association",
        ],
    ):
        draws = {
            str(feature): values.sel({feature_dim: feature}).values
            for feature in features
        }
        plot_intervals(ax, draws)
        ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, output)


def plot_h4(idata, output):
    """H4: Preference change across natural-hazard and financial vulnerability."""
    posterior = idata.posterior

    shift_mean = posterior["shift_mean"].mean(("chain", "draw"))
    shift_nh = posterior["shift_nh"].mean(("chain", "draw"))
    shift_fin = posterior["shift_fin"].mean(("chain", "draw"))

    feature_dim = next(
        dim for dim in shift_mean.dims
        if dim not in {"chain", "draw"}
    )
    features = shift_mean.coords[feature_dim].values

    nh_values = np.linspace(-2, 2, 41)
    fv_values = np.linspace(-2, 2, 41)

    n_features = len(features)
    ncols = 2
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11, 4 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    image = None

    for ax, feature in zip(axes, features):
        intercept = float(shift_mean.sel({feature_dim: feature}))
        beta_nh = float(shift_nh.sel({feature_dim: feature}))
        beta_fv = float(shift_fin.sel({feature_dim: feature}))

        surface = (
            intercept
            + beta_nh * nh_values[None, :]
            + beta_fv * fv_values[:, None]
        )

        image = ax.imshow(
            surface,
            origin="lower",
            aspect="auto",
            extent=[
                nh_values.min(),
                nh_values.max(),
                fv_values.min(),
                fv_values.max(),
            ],
        )

        ax.set_title(str(feature))
        ax.set_xlabel("Natural-hazard vulnerability")
        ax.set_ylabel("Financial vulnerability")

    for ax in axes[n_features:]:
        ax.axis("off")

    if image is not None:
        fig.colorbar(
            image,
            ax=axes[:n_features].tolist(),
            label="Predicted preference change",
            shrink=0.8,
        )

    fig.tight_layout()
    save_figure(fig, output)

 
def main(snakemake) -> None:
    idata = az.from_netcdf(snakemake.input.model)

    plotters = {
        "H1": lambda idata, output: plot_h1(
            idata,
            output,
            snakemake.params.conjoint,
        ),
        "H2": plot_h2,
        "H3": plot_h3,
        "H4": plot_h4,
    }

    for figure in snakemake.params.figures:
        plotters[figure["hypothesis"]](idata, figure["output"])


if __name__ == "__main__":
    main(globals()["snakemake"])