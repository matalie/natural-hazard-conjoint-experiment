rule pre_post_distributions:
    input:
        analysis=ANALYSIS,
        code=[SRC+"plotting.py", SRC+"descriptive_plots.py"],
    output:
        figure=f"{ROOT}/figures/{{distribution}}.png"
    params:
        kind="pre_post",
        figure=lambda wc: DISTRIBUTIONS[wc.distribution],
    wildcard_constraints:
        distribution=regex_keys(DISTRIBUTIONS)
    conda: "../envs/environment.yaml"
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
    conda: "../envs/environment.yaml"
    script: "../scripts/plot_distributions.py"


rule likert_shifts:
    input:
        analysis=ANALYSIS,
        code=[SRC + "descriptive_plots.py", SRC + "plotting.py"],
    output:
        figure=LIKERT_SHIFTS_FIGURE
    params:
        kind="likert_shifts"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_distributions.py"


rule plot_hypothesis:
    input:
        model=f"{ROOT}/models/main.nc",
        code=[SRC + "posterior.py", SRC + "plotting.py", SRC + "hypothesis_plots.py",],
    output:
        figure=f"{ROOT}/figures/hypotheses/{{hypothesis_plot}}.png",
        table=f"{ROOT}/tables/hypotheses/{{hypothesis_plot}}.csv",
    params:
        figure=lambda wc: HYPOTHESES[wc.hypothesis_plot],
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    wildcard_constraints:
        hypothesis_plot=regex_keys(HYPOTHESES)
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_hypotheses.py"


rule efa:
    input:
        analysis=ANALYSIS,
        code=[SRC+"efa.py", SRC+"plotting.py"],
    output:
        correlation=EFA_CORRELATION_FIGURE,
        efa=EFA_COMPARISON_FIGURE,
        correlation_table=f"{ROOT}/tables/efa/correlation_matrix.csv",
        loadings2=f"{ROOT}/tables/efa/loadings_2_factors.csv",
        loadings3=f"{ROOT}/tables/efa/loadings_3_factors.csv",
        phi2=f"{ROOT}/tables/efa/phi_2_factors.csv",
        phi3=f"{ROOT}/tables/efa/phi_3_factors.csv",
    conda: "../envs/efa.yaml"
    script: "../scripts/efa.py"


rule plot_respondents:
    input:
        model=f"{ROOT}/models/main.nc",
        code=[SRC+"posterior.py", SRC+"respondent_plots.py", SRC+"plotting.py"],
    output:
        latent=LATENT_CONSTRUCT_FIGURE,
        impacts=RESPONDENT_IMPACT_FIGURES,
        tables=RESPONDENT_IMPACT_TABLES,
    params:
        specifications=RESPONDENT_IMPACT_SPECS,
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    conda: "../envs/environment.yaml"
    script: "../scripts/plot_respondents.py"


rule plot_diagnostics:
    input:
        model=f"{ROOT}/models/main.nc",
        code=[
            SRC + "diagnostic_plots.py",
            SRC + "posterior.py",
            SRC + "plotting.py",
        ],
    output:
        figure=TRACE_FIGURE,
    params:
        conjoint=config["conjoint"],
        variable="shift_mean",
    conda: "../envs/environment.yaml"
    script: "../scripts/plot_diagnostics.py"

rule analyze_robustness:
    input:
        models=lambda wc: [f"{ROOT}/models/{run}.nc" for run in ROBUSTNESS[wc.comparison]["runs"]],
        conjoint=CONJOINT,
        code=[
            SRC+"posterior.py",
            SRC+"plotting.py",
            SRC+"robustness_analysis.py",
            SRC+"data_preparation.py",
        ],
    output:
        figure=f"{ROOT}/figures/robustness/{{comparison}}.png",
        table=f"{ROOT}/tables/robustness/{{comparison}}.csv",
        audit=f"{ROOT}/tables/robustness/{{comparison}}-sample-audit.csv",
    params:
        specification=lambda wc: ROBUSTNESS[wc.comparison],
        runs=config["models"]["runs"],
        samples=config["samples"],
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    wildcard_constraints:
        comparison=regex_keys(ROBUSTNESS)
    conda: "../envs/environment.yaml"
    script: "../scripts/analyze_robustness.py"


rule assign_geography:
    input:
        respondents=COMBINED,
        boundaries=GEO["boundaries"],
        localities=GEO["localities"],
        overrides=([GEO["overrides"]] if GEO.get("overrides") else []),
        code=[SRC+"maps.py"],
    output:
        assignments=ASSIGNMENTS,
        audit=f"{ROOT}/tables/audit/geographic_assignment.csv",
    params:
        stage="assign",
        geography=GEO,
    conda: "../envs/geo.yaml"
    script: "../scripts/maps.py"


rule plot_survey_map:
    input:
        respondents=ANALYSIS if config["maps"]["survey_source"] == "analysis" else COMBINED,
        assignments=ASSIGNMENTS,
        boundaries=GEO["boundaries"],
        code=[SRC+"maps.py", SRC+"plotting.py"],
    output:
        figure=f"{ROOT}/figures/maps/survey-{{map_name}}.png",
        table=f"{ROOT}/tables/maps/survey-{{map_name}}.csv",
    params:
        stage="survey",
        figure=lambda wc: SURVEY_MAPS[wc.map_name],
        geography=GEO,
    wildcard_constraints:
        map_name=regex_keys(SURVEY_MAPS)
    conda: "../envs/geo.yaml"
    script: "../scripts/maps.py"


rule plot_posterior_map:
    input:
        model=lambda wc: f"{ROOT}/models/{POSTERIOR_MAPS[wc.map_name].get('run', 'main')}.nc",
        respondents=ANALYSIS,
        assignments=ASSIGNMENTS,
        boundaries=GEO["boundaries"],
        code=[SRC+"maps.py", SRC+"posterior.py", SRC+"plotting.py"],
    output:
        figure=f"{ROOT}/figures/maps/posterior-{{map_name}}.png",
        table=f"{ROOT}/tables/maps/posterior-{{map_name}}.csv",
    params:
        stage="posterior",
        figure=lambda wc: POSTERIOR_MAPS[wc.map_name],
        geography=GEO,
        conjoint=config["conjoint"],
        hdi_prob=HDI,
    wildcard_constraints:
        map_name=regex_keys(POSTERIOR_MAPS)
    conda: "../envs/geo.yaml"
    script: "../scripts/maps.py"
