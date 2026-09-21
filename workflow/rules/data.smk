rule clean_surveys:
    input:
        s0=config["data"]["raw"]["S0"],
        s1=config["data"]["raw"]["S1"],
        ids=config["data"]["id_mapping"]["path"],
        code=[SRC+"data_preparation.py", SRC+"mappings.py"],
    output:
        s0=f"{ROOT}/data/waves/S0_clean.parquet",
        s1=f"{ROOT}/data/waves/S1_clean.parquet",
        combined=COMBINED,
        report=f"{ROOT}/tables/audit/cleaning.json",
    params:
        stage="clean",
        preprocessing={k: v for k, v in config["preprocessing"].items() if k not in {"imputation", "acceptance_consistency"}},
        id_mapping=config["data"]["id_mapping"],
    conda: "../envs/environment.yaml"
    script: "../scripts/preprocess.py"


rule analysis_sample:
    input:
        combined=COMBINED,
        code=[SRC+"data_preparation.py", SRC+"mappings.py"],
    output:
        analysis=ANALYSIS,
        audit=f"{ROOT}/tables/audit/imputation_by_respondent.csv",
    params:
        stage="sample",
        analysis_items=config["analysis_items"],
        imputation=config["preprocessing"]["imputation"],
        acceptance=config["preprocessing"]["acceptance_consistency"],
    conda: "../envs/environment.yaml"
    script: "../scripts/preprocess.py"


rule conjoint_data:
    input:
        analysis=ANALYSIS,
        code=[SRC+"data_preparation.py", SRC+"mappings.py"],
    output: conjoint=CONJOINT
    params:
        stage="conjoint",
        conjoint=config["conjoint"],
    conda: "../envs/environment.yaml"
    script: "../scripts/preprocess.py"
