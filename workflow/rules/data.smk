"""Survey cleaning, analysis-sample construction, and conjoint data preparation."""

CLEANING_CONFIG = {
    key: value
    for key, value in config["preprocessing"].items()
    if key not in {"imputation", "acceptance_consistency"}
}


# Clean and link the two survey waves
# -----------------------------------
rule clean_surveys:
    input:
        s0=config["data"]["raw"]["S0"],
        s1=config["data"]["raw"]["S1"],
        ids=config["data"]["id_mapping"]["path"],
        code=[SRC+"data_preparation.py", SRC+"mappings.py"],
    output:
        s0=f"{DATA_DIR}/waves/S0_clean.parquet",
        s1=f"{DATA_DIR}/waves/S1_clean.parquet",
        combined=COMBINED,
        report=f"{TABLE_DIR}/audit/cleaning.json",
    params:
        stage="clean",
        preprocessing=CLEANING_CONFIG,
        id_mapping=config["data"]["id_mapping"],
    script: "../scripts/preprocess.py"


# Construct the final respondent sample, apply configured imputation
# ------------------------------------------------------------------
rule analysis_sample:
    input:
        combined=COMBINED,
        code=[SRC+"data_preparation.py", SRC+"mappings.py"],
    output:
        analysis=ANALYSIS,
        audit=f"{TABLE_DIR}/audit/imputation_by_respondent.csv",
    params:
        stage="sample",
        analysis_items=config["analysis_items"],
        imputation=config["preprocessing"]["imputation"],
        acceptance=config["preprocessing"]["acceptance_consistency"],
    script: "../scripts/preprocess.py"


# Transform respondent sample into conjoint design
# ------------------------------------------------
rule conjoint_data:
    input:
        analysis=ANALYSIS,
        code=[SRC+"data_preparation.py", SRC+"mappings.py"],
    output:
        conjoint=CONJOINT,
    params:
        stage="conjoint",
        conjoint=config["conjoint"],
    script: "../scripts/preprocess.py"