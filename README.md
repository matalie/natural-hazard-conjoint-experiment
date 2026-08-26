# Natural Hazard Solidarity – Snakemake workflow

This is a compact rewrite of the old notebook-based repository. The reproducible workflow no longer depends on notebooks.

## What the workflow does

`resources/raw/*` → clean S0/S1 → link both waves → impute required HCM items → create long conjoint data → effect/dummy coding → pre/post distribution plots → fit all enabled HCM runs.

Generated files include:

- `results/data/waves/S0_clean.parquet`
- `results/data/waves/S1_clean.parquet`
- `results/data/combined_surveys.parquet`
- `results/data/analysis_sample.parquet`
- `results/data/conjoint_effect.parquet` or `conjoint_dummy.parquet`
- `results/figures/distribution-climate-likert.png`
- `results/figures/distribution-nh-likert.png`
- `results/models/<run>.nc`

## Files kept deliberately small in number

- `config/config.yaml`: paths, preprocessing decisions, conjoint coding, samples, model runs, sampling settings, plots.
- `workflow/Snakefile`: the complete DAG. No separate `rules/*.smk` files yet.
- `workflow/scripts/preprocess.py`: one coherent preprocessing stage that writes all relevant Parquet checkpoints.
- `workflow/scripts/fit_model.py`: generic runner for every configured model run.
- `workflow/scripts/plot_distributions.py`: only the early pre/post distribution plots requested for now.
- `src/natural_hazard_solidarity/data.py`: reusable cleaning/conjoint/sample logic.
- `src/natural_hazard_solidarity/models.py`: the two genuinely different HCM structures.
- `src/natural_hazard_solidarity/mappings.py`: survey response/translation mappings from the old repo.
- `workflow/envs/environment.yaml`: one environment for now; split it later only if there is a real need.

## Raw files

Put the local raw files under `resources/raw/` using the names configured in `config/config.yaml`. `resources/`, `results/`, and `notebooks/` are ignored by Git.

Check the ID-linking columns in `config.yaml` (`S0_idx`, `S1_idx`, `id`, `m`) against the real `id_list.csv` before the first run.

## Run

From the repository root, in the Miniforge Prompt:

```cmd
conda activate snakemake
snakemake -n
snakemake --cores 1 --sdm conda
```

`--cores 1` is a conservative Windows default because the old notebooks explicitly noted memory crashes during parallel PyMC sampling. PyMC itself still runs the configured four chains serially because `sampling.cores: 1`.

To fit only the main model while developing:

```cmd
snakemake --cores 1 --sdm conda results/models/main.nc
```

To run preprocessing only:

```cmd
snakemake --cores 1 --sdm conda results/data/conjoint_effect.parquet
```

If `conjoint.coding` is changed to `dummy`, the target becomes `results/data/conjoint_dummy.parquet`.

## Model consolidation

The duplicated `complete_longitudinal_hcm*` notebooks are represented by config entries rather than copied model code. `static` uses the final easy-factor structure; `adjusted` keeps the genuinely different pre/change NHV and financial-vulnerability structure.

The config includes the completed robustness runs from the old repo: prior factors 1/2/4, without FV, German-speaking sample, age 35+, non-low-income sample, acceptance-consistent sample, and the company-task exclusion. Disable any run with `enabled: false`.

One intentional correction: the old “wo company” model notebook loaded the full `conjoint_df.csv` even though `conjoint_wo_company_df.csv` had been created. This workflow applies the intended task exclusion before fitting `without_company`.

The acceptance-consistency rule also documents an old naming ambiguity: `all_acceptance_consistent` actually allowed up to two inconsistent tasks. The workflow preserves that behavior via `max_inconsistent_tasks: 2`.

## Not migrated on purpose

Exploratory latent-construct notebooks, respondent exploration, older choice-model prototypes, and the large posterior-analysis notebooks are not part of this first compact workflow. Their final reusable analyses can be added later without changing preprocessing or model training.
