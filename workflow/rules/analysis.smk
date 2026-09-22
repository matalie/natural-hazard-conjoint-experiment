"""Rules for analysis, plotting, robustness, and map plots."""

# -----------------------
# Configuration
# -----------------------
HYPOTHESES = config["plots"]["hypotheses"]
DISTRIBUTIONS = config["plots"]["pre_post_distributions"]
ROBUSTNESS = config["analysis"]["robustness"]
SURVEY_MAPS = config["maps"]["survey"]
POSTERIOR_MAPS = config["maps"]["posterior"]
GEO = config["geography"]

# -----------------------
# Output paths
# -----------------------
HYPOTHESIS_FIGURE_DIR = f"{FIGURE_DIR}/hypotheses"
ROBUSTNESS_FIGURE_DIR = f"{FIGURE_DIR}/robustness"
MAP_FIGURE_DIR = f"{FIGURE_DIR}/maps"

HYPOTHESIS_FIGURES = [f"{HYPOTHESIS_FIGURE_DIR}/{name}.png" for name in HYPOTHESES]
DISTRIBUTION_FIGURES = [f"{FIGURE_DIR}/{name}.png" for name in DISTRIBUTIONS]
ROBUSTNESS_FIGURES = [f"{ROBUSTNESS_FIGURE_DIR}/{name}.png" for name in ROBUSTNESS]
SURVEY_MAP_FIGURES = [f"{MAP_FIGURE_DIR}/survey-{name}.png" for name in SURVEY_MAPS]
POSTERIOR_MAP_FIGURES = [f"{MAP_FIGURE_DIR}/posterior-{name}.png" for name in POSTERIOR_MAPS]

DESERVINGNESS_FIGURE = f"{FIGURE_DIR}/likert_histograms_deservingness.png"
LIKERT_SHIFTS_FIGURE = f"{FIGURE_DIR}/Likert-shifts.png"
EFA_CORRELATION_FIGURE = f"{FIGURE_DIR}/Correlation-matrix.png"
EFA_COMPARISON_FIGURE = f"{FIGURE_DIR}/EFA-2-and-3-factors-comparison-square-cells.png"
LATENT_CONSTRUCT_FIGURE = f"{FIGURE_DIR}/latent_constructs_eta_pd_eta_nhv.png"
TRACE_FIGURE = f"{FIGURE_DIR}/trace_posterior_shift_mean_all_groups.png"

EFA_FIGURES = [EFA_CORRELATION_FIGURE, EFA_COMPARISON_FIGURE]

RESPONDENT_IMPACT_SPECS = [
    {"attribute": attr, "level": level, "index": i}
    for attr, spec in config["conjoint"]["attributes"].items()
    for i, level in enumerate(spec["levels"], start=1)
]

RESPONDENT_IMPACT_FIGURES = [
    f"{FIGURE_DIR}/respondent-impact-{s['attribute']}-{s['index']:02d}.png"
    for s in RESPONDENT_IMPACT_SPECS
]

RESPONDENT_IMPACT_TABLES = [
    f"{TABLE_DIR}/respondent_impact/{s['attribute']}-{s['index']:02d}.csv"
    for s in RESPONDENT_IMPACT_SPECS
]

REQUESTED_FIGURES = (
    DISTRIBUTION_FIGURES
    + [DESERVINGNESS_FIGURE, LIKERT_SHIFTS_FIGURE]
    + EFA_FIGURES
    + [LATENT_CONSTRUCT_FIGURE]
    + RESPONDENT_IMPACT_FIGURES
    + [TRACE_FIGURE]
)

ALL_MAP_FIGURES = SURVEY_MAP_FIGURES + POSTERIOR_MAP_FIGURES

# Tables
MAIN_PARAMETER_TABLE = f"{TABLE_DIR}/diagnostics/main-model-parameters.csv"
LATENT_LOADING_TABLE = f"{TABLE_DIR}/diagnostics/latent-loadings.csv"

MODEL_DIAGNOSTIC_TABLES = [MAIN_PARAMETER_TABLE, LATENT_LOADING_TABLE,]

# ---------------------------------
# validate robustness configuration
# ---------------------------------
for name, spec in ROBUSTNESS.items():
    runs = spec["runs"]
    if len(runs) != len(set(runs)):
        raise ValueError(f"Duplicate model in robustness comparison '{name}'.")
    undefined = set(runs) - set(RUNS)
    if undefined:
        raise ValueError(
            f"Robustness comparison '{name}' references undefined run(s): {sorted(undefined)}"
        )

# ---------------------------------
# # Plotting rules
# ---------------------------------

# Survey data plots
# -----------------
rule pre_post_distributions:
    input:
        analysis=ANALYSIS,
        code=[SRC+"plotting.py", SRC+"descriptive_plots.py"],
    output:
        figure=f"{FIGURE_DIR}/{{distribution}}.png"
    params:
        kind="pre_post",
        figure=lambda wc: DISTRIBUTIONS[wc.distribution],
    wildcard_constraints:
        distribution=regex_keys(DISTRIBUTIONS)
    script: "../scripts/plot_distributions.py"


rule deservingness_distribution:
    input:
        analysis=ANALYSIS,
        code=[SRC+"plotting.py", SRC+"descriptive_plots.py"],
    output:
        figure=DESERVINGNESS_FIGURE
    params:
        kind="deservingness",
        figure={},
    script: "../scripts/plot_distributions.py"

rule likert_shifts:
    input:
        analysis=ANALYSIS,
        code=[SRC+"descriptive_plots.py", SRC+"plotting.py"],
    output:
        figure=LIKERT_SHIFTS_FIGURE
    params:
        kind="likert_shifts"
    script: "../scripts/plot_distributions.py"

# EFA
# ---
rule plot_efa:
    input:
        analysis=ANALYSIS,
        code=[SRC+"efa.py", SRC+"plotting.py"],
    output:
        correlation=EFA_CORRELATION_FIGURE,
        comparison=EFA_COMPARISON_FIGURE,
        correlation_table=f"{TABLE_DIR}/efa/correlation_matrix.csv",
        loadings_2=f"{TABLE_DIR}/efa/loadings_2_factors.csv",
        loadings_3=f"{TABLE_DIR}/efa/loadings_3_factors.csv",
        phi_2=f"{TABLE_DIR}/efa/phi_2_factors.csv",
        phi_3=f"{TABLE_DIR}/efa/phi_3_factors.csv",
    script:
        "../scripts/efa.py"

# Model posterior plots
# ---------------------
rule plot_hypothesis:
    input:
        model=f"{MODEL_DIR}/main.nc",
        code=[SRC+"posterior.py", SRC+"plotting.py", SRC+"hypothesis_plots.py"],
    output:
        figure=f"{HYPOTHESIS_FIGURE_DIR}/{{hypothesis_plot}}.png",
        table=f"{TABLE_DIR}/hypotheses/{{hypothesis_plot}}.csv",
    params:
        figure=lambda wc: HYPOTHESES[wc.hypothesis_plot],
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    wildcard_constraints:
        hypothesis_plot=regex_keys(HYPOTHESES)
    script: "../scripts/plot_hypotheses.py"

rule plot_respondents:
    input:
        model=f"{MODEL_DIR}/main.nc",
        code=[SRC+"posterior.py", SRC+"respondent_plots.py", SRC+"plotting.py"],
    output:
        latent=LATENT_CONSTRUCT_FIGURE,
        impacts=RESPONDENT_IMPACT_FIGURES,
        tables=RESPONDENT_IMPACT_TABLES,
    params:
        specifications=RESPONDENT_IMPACT_SPECS,
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    script: "../scripts/plot_respondents.py"

rule plot_diagnostics:
    input:
        model=f"{MODEL_DIR}/main.nc",
        code=[SRC+"diagnostic_plots.py", SRC+"posterior.py", SRC+"plotting.py"],
    output:
        figure=TRACE_FIGURE
    params:
        conjoint=config["conjoint"],
        variable="shift_mean",
    script: "../scripts/plot_diagnostics.py"

rule model_diagnostic_tables:
    input:
        model=f"{MODEL_DIR}/main.nc",
        code=SRC+"diagnostic_tables.py",
    output:
        main=MAIN_PARAMETER_TABLE,
        latent=LATENT_LOADING_TABLE,
    script:
        "../scripts/export_model_diagnostics.py"


# Robustness plots
# ----------------
rule analyze_robustness:
    input:
        models=lambda wc: [
            f"{MODEL_DIR}/{run}.nc"
            for run in ROBUSTNESS[wc.comparison]["runs"]
        ],
        conjoint=CONJOINT,
        code=[
            SRC+"posterior.py",
            SRC+"plotting.py",
            SRC+"robustness_analysis.py",
            SRC+"data_preparation.py",
        ],
    output:
        figure=f"{ROBUSTNESS_FIGURE_DIR}/{{comparison}}.png",
        table=f"{TABLE_DIR}/robustness/{{comparison}}.csv",
        audit=f"{TABLE_DIR}/robustness/{{comparison}}-sample-audit.csv",
    params:
        specification=lambda wc: ROBUSTNESS[wc.comparison],
        runs=RUNS,
        samples=config["samples"],
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    wildcard_constraints:
        comparison=regex_keys(ROBUSTNESS)
    script: "../scripts/analyze_robustness.py"

# Maps plots and analysis
# -----------------------
rule assign_geography:
    input:
        respondents=COMBINED,
        boundaries=GEO["boundaries"],
        localities=GEO["localities"],
        code=SRC+"maps.py",
    output:
        assignments=ASSIGNMENTS,
        audit=f"{TABLE_DIR}/audit/geographic_assignment.csv",
    params:
        stage="assign",
        geography=GEO,
    script: "../scripts/maps.py"

rule plot_survey_map:
    input:
        respondents=ANALYSIS if config["maps"]["survey_source"] == "analysis" else COMBINED,
        assignments=ASSIGNMENTS,
        boundaries=GEO["boundaries"],
        code=[SRC+"maps.py", SRC+"plotting.py"],
    output:
        figure=f"{MAP_FIGURE_DIR}/survey-{{map_name}}.png",
        table=f"{TABLE_DIR}/maps/survey-{{map_name}}.csv",
    params:
        stage="survey",
        figure=lambda wc: SURVEY_MAPS[wc.map_name],
        geography=GEO,
    wildcard_constraints:
        map_name=regex_keys(SURVEY_MAPS)
    script: "../scripts/maps.py"

rule plot_posterior_map:
    input:
        model=lambda wc: f"{MODEL_DIR}/{POSTERIOR_MAPS[wc.map_name].get('run', 'main')}.nc",
        respondents=ANALYSIS,
        assignments=ASSIGNMENTS,
        boundaries=GEO["boundaries"],
        code=[SRC+"maps.py", SRC+"posterior.py", SRC+"plotting.py"],
    output:
        figure=f"{MAP_FIGURE_DIR}/posterior-{{map_name}}.png",
        table=f"{TABLE_DIR}/maps/posterior-{{map_name}}.csv",
    params:
        stage="posterior",
        figure=lambda wc: POSTERIOR_MAPS[wc.map_name],
        geography=GEO,
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    wildcard_constraints:
        map_name=regex_keys(POSTERIOR_MAPS)
    script: "../scripts/maps.py"


# --------------------------
# Public convenience targets
# --------------------------
rule hypotheses:
    input:
        HYPOTHESIS_FIGURES

rule requested_plots:
    input:
        REQUESTED_FIGURES

rule robustness_plots:
    input:
        ROBUSTNESS_FIGURES

rule maps:
    input:
        ALL_MAP_FIGURES

rule diagnostics:
    input:
        TRACE_FIGURE,
        MODEL_DIAGNOSTIC_TABLES