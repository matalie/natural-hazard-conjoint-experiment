rule fit_model:
    input:
        conjoint=CONJOINT,
        code=[SRC+"models.py", SRC+"data_preparation.py", SRC+"mappings.py"],
    output: model=f"{ROOT}/models/{{run}}.nc"
    params:
        run_name=lambda wc: wc.run,
        run=lambda wc: RUNS[wc.run],
        sample=lambda wc: config["samples"][RUNS[wc.run]["sample"]],
        sampling=config["sampling"],
        conjoint=config["conjoint"],
        constructs=config["constructs"],
    threads: min(config["sampling"]["cores"], config["sampling"]["chains"])
    wildcard_constraints: run=regex_keys(RUNS)
    conda: "../envs/environment.yaml"
    script: "../scripts/fit_model.py"


rule compare_models:
    input:
        models=lambda wc: [f"{ROOT}/models/{name}.nc" for name in COMPARISONS[wc.comparison]],
        code=SRC+"posterior.py",
    output:
        table=f"{ROOT}/tables/comparisons/{{comparison}}.csv",
        pareto=f"{ROOT}/tables/comparisons/{{comparison}}-pareto.csv",
    params: runs=lambda wc: COMPARISONS[wc.comparison]
    wildcard_constraints: comparison=regex_keys(COMPARISONS)
    conda: "../envs/environment.yaml"
    script: "../scripts/compare_models.py"
