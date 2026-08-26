WAVES = ["S0", "S1"]
CODING = config["conjoint"]["coding"]

rule clean_wave:
    input:
        raw=lambda wc: config["data"]["raw"][wc.wave]
    output:
        cleaned="results/data/waves/{wave}_clean.parquet"
    params:
        skiprows=config["preprocessing"]["skiprows"],
        preprocessing=config["preprocessing"]
    wildcard_constraints:
        wave="S0|S1"
    script:
        "../scripts/clean_wave.py"

rule merge_waves:
    input:
        s0="results/data/waves/S0_clean.parquet",
        s1="results/data/waves/S1_clean.parquet",
        ids=config["data"]["id_mapping"]["path"]
    output:
        merged="results/data/combined_surveys.parquet"
    params:
        id_mapping=config["data"]["id_mapping"]
    script:
        "../scripts/merge_waves.py"

rule prepare_analysis_sample:
    input:
        merged="results/data/combined_surveys.parquet"
    output:
        prepared="results/data/analysis_sample.parquet"
    params:
        items=config["analysis_items"],
        imputation=config["preprocessing"]["imputation"]
    script:
        "../scripts/prepare_analysis_sample.py"

rule build_conjoint:
    input:
        prepared="results/data/analysis_sample.parquet"
    output:
        conjoint=f"results/data/conjoint_{CODING}.parquet"
    params:
        conjoint=config["conjoint"]
    script:
        "../scripts/build_conjoint.py"