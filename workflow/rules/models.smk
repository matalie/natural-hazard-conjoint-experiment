"""Model fitting and model-comparison rules."""

# -----------------------
# Configuration
# -----------------------

COMPARISONS = config["analysis"]["comparisons"]

DEFAULT_RUNS = config["models"]["default_runs"]
DEFAULT_MODEL_FILES = [f"{MODEL_DIR}/{run}.nc" for run in DEFAULT_RUNS]

MODEL_COMPARISON_TABLES = [f"{TABLE_DIR}/comparisons/{name}.csv" for name in COMPARISONS]

# ---------------------------------
# Model fitting and comparison
# ---------------------------------

# existence check
# ---------------
undefined_defaults = set(DEFAULT_RUNS) - set(RUNS)
if undefined_defaults:
    raise ValueError(
        f"default_runs references undefined run(s): {sorted(undefined_defaults)}"
    )

for name, runs in COMPARISONS.items():
    undefined = set(runs) - set(RUNS)
    if undefined:
        raise ValueError(
            f"LOO comparison '{name}' references undefined run(s): {sorted(undefined)}"
        )


# Model fitting
# -------------
rule fit_model:
    input:
        conjoint=CONJOINT,
        code=[SRC+"models.py", SRC+"data_preparation.py", SRC+"mappings.py"],
    output:
        model=f"{MODEL_DIR}/{{run}}.nc"
    params:
        run_name=lambda wc: wc.run,
        run=lambda wc: RUNS[wc.run],
        sample=lambda wc: config["samples"][RUNS[wc.run]["sample"]],
        sampling=config["sampling"],
        conjoint=config["conjoint"],
        constructs=config["constructs"],
    threads:
        min(config["sampling"]["cores"], config["sampling"]["chains"])
    wildcard_constraints:
        run=regex_keys(RUNS)
    script: "../scripts/fit_model.py"


# LOO model comparison
# --------------------
rule compare_models:
    input:
        models=lambda wc: [
            f"{MODEL_DIR}/{run}.nc"
            for run in COMPARISONS[wc.comparison]
        ],
        code=SRC+"posterior.py",
    output:
        table=f"{TABLE_DIR}/comparisons/{{comparison}}.csv",
        pareto=f"{TABLE_DIR}/comparisons/{{comparison}}-pareto.csv",
    params:
        runs=lambda wc: COMPARISONS[wc.comparison]
    wildcard_constraints:
        comparison=regex_keys(COMPARISONS)
    script: "../scripts/compare_models.py"


# ---------------------------------
# Public convenience targets
# ---------------------------------
rule main_model:
    input:
        f"{MODEL_DIR}/main.nc"

rule model_comparison:
    input: MODEL_COMPARISON_TABLES