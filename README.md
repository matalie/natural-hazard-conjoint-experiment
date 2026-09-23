# Natural Hazard Solidarity

Reproducible Snakemake workflow for the two-wave natural-hazard solidarity survey and conjoint experiment.

The workflow covers:

- survey preprocessing and sample construction
- conjoint encoding
- Bayesian model estimation
- hypothesis analyses
- descriptive and respondent-level figures
- exploratory factor analysis
- robustness analyses
- geographic analyses and maps
- model diagnostics and comparison

---

## Quick start

### Requirements

The workflow is currently designed for Windows and requires:

1. [Git](https://git-scm.com/)
2. [Miniforge](https://github.com/conda-forge/miniforge)
3. [MSYS2](https://www.msys2.org/) with the UCRT64 GCC compiler

The Python environment and all Python/R-independent workflow dependencies are defined in:

```text
environment.yaml
```

---

## 1. Clone the repository

Open a terminal and clone the repository:

```cmd
git clone <repository-url>
cd natural_hazard_solidarity
```

---

## 2. Create the Conda environment

Create the project environment from `environment.yaml`:

```cmd
conda env create -f environment.yaml
```

The default environment name is:

```text
natural-hazard-solidarity
```

You can verify that the environment was created with:

```cmd
conda env list
```

---

## 3. Configure local settings

Machine-specific settings are stored separately from the repository.

Copy:

```text
local_settings.example.bat
```

to:

```text
local_settings.bat
```

For example:

```cmd
copy local_settings.example.bat local_settings.bat
```

Then open `local_settings.bat` and adjust the paths if necessary.

A typical configuration is:

```bat
@echo off

set "PROJECT_ENV=natural-hazard-solidarity"
set "SNAKEMAKE_CORES=4"

set "MSYS2_UCRT64=C:\msys64\ucrt64"
set "PYTENSOR_CXX=C:/msys64/ucrt64/bin/g++.exe"
```

`local_settings.bat` is intentionally excluded from Git because it contains machine-specific paths.

---

## 4. Add the required input data

The raw survey and geographic input data are not distributed with this repository.

Place the required files in:

```text
resources/
```

The expected structure is approximately:

```text
resources/
├── Survey_Adaptation Natural Hazards_First Wave_raw data.csv
├── Survey_Adaptation Natural Hazards_Second Wave_raw data.csv
├── id_list.csv
├── swissBOUNDARIES3D_1_5_LV95_LN02.gpkg
└── AMTOVZ_CSV_WGS84.csv
```

See:

```text
docs/data.md
```

for a complete description of the required input files, their purpose, and their source.

---

## 5. Test the installation

Before running the full workflow, perform a Snakemake dry run:

```cmd
run_workflow.bat -n
```

This checks whether Snakemake can construct the workflow DAG without executing any jobs.

Then run the automated project checks:

```cmd
run_workflow.bat check
```

If both commands complete successfully, the installation is ready.

---

## 6. Run the complete workflow

To execute the complete default workflow:

```cmd
run_workflow.bat
```

Snakemake automatically determines which files are missing or outdated and runs only the required steps.

Already up-to-date outputs are not recomputed.

A successful fully up-to-date workflow should end with a message similar to:

```text
Nothing to be done.
```

---

## Running individual workflow components

Individual parts of the analysis can be executed separately.

### Preprocessing

```cmd
run_workflow.bat preprocess
```

Processes the raw survey data, combines survey waves, constructs the analysis sample, and prepares the conjoint data.

Typical outputs are stored under:

```text
results/data/
```

---

### Main Bayesian model

```cmd
run_workflow.bat main_model
```

Fits the main hierarchical conjoint model.

Main output:

```text
results/models/main.nc
```

The NetCDF file contains the posterior draws and model metadata required by downstream analyses.

---

### Hypothesis analyses

```cmd
run_workflow.bat hypotheses
```

Generates the main hypothesis figures and corresponding numerical result tables.

Typical outputs:

```text
results/figures/hypotheses/
├── H1-pre-post-shift.png
├── H2-psychological-distance.png
├── H3-financial-vulnerability.png
└── H4-vulnerability-heatmap.png
```

and:

```text
results/tables/hypotheses/
```

---

### Requested and descriptive plots

```cmd
run_workflow.bat requested_plots
```

Generates the configured descriptive, exploratory, latent-construct, respondent-level, and diagnostic figures.

Typical outputs are stored under:

```text
results/figures/
```

and, where applicable:

```text
results/tables/
```

---

### Model diagnostics

```cmd
run_workflow.bat diagnostics
```

Generates diagnostic summaries for the main model, including posterior parameter summaries and sampling diagnostics.

Typical outputs:

```text
results/tables/diagnostics/
├── main-model-parameters.csv
└── latent-loadings.csv
```

Diagnostic figures may additionally be stored under:

```text
results/figures/
```

---

### Maps

```cmd
run_workflow.bat maps
```

Generates survey-based and posterior geographic maps.

Typical outputs:

```text
results/figures/maps/
```

with corresponding numerical data under:

```text
results/tables/maps/
```

---

### Robustness analyses

```cmd
run_workflow.bat robustness_plots
```

Runs the robustness-model analyses required for the configured robustness figures and produces comparison plots and tables.

Typical outputs:

```text
results/figures/robustness/
```

and:

```text
results/tables/robustness/
```

---

### Model comparison

```cmd
run_workflow.bat model_comparison
```

Computes the configured model comparisons, including LOO-based comparisons where applicable.

Typical outputs:

```text
results/tables/comparisons/
```

---

## Workflow overview

The main workflow follows approximately this dependency structure:

```text
raw survey data
       │
       ▼
 preprocessing
       │
       ├──────────────► descriptive analyses
       │
       ▼
 analysis sample
       │
       ▼
 conjoint encoding
       │
       ▼
 Bayesian main model
       │
       ├──────────────► hypothesis analyses
       │
       ├──────────────► posterior diagnostics
       │
       ├──────────────► respondent-level analyses
       │
       ├──────────────► geographic analyses
       │
       └──────────────► robustness / model comparison
```

Snakemake tracks these dependencies automatically.

If an upstream input or source file changes, the affected downstream outputs are regenerated.

---

## Configuration

Scientific and workflow settings are defined in:

```text
config/config.yaml
```

This file contains project-level settings that should be reproducible across machines.

Machine-specific settings instead belong in:

```text
local_settings.bat
```

Examples include:

- Conda environment name
- number of Snakemake cores
- local MSYS2 path
- local C++ compiler path

Do not place machine-specific paths in `config/config.yaml`.

---

## Output structure

Generated files are written below:

```text
results/
```

The main structure is approximately:

```text
results/
├── data/
├── figures/
│   ├── hypotheses/
│   ├── maps/
│   └── robustness/
├── models/
└── tables/
    ├── audit/
    ├── diagnostics/
    ├── hypotheses/
    ├── maps/
    ├── robustness/
    └── comparisons/
```

The `results/` directory is generated by the workflow and is not intended to contain manually edited source data.

---

## Re-running the workflow

Snakemake only executes jobs whose outputs are:

- missing
- older than relevant input files
- affected by changed dependencies

For example, if:

```text
results/figures/hypotheses/H1-pre-post-shift.png
```

is deleted, running:

```cmd
run_workflow.bat
```

will regenerate the missing figure without unnecessarily rerunning unrelated analyses.

A dry run can be used to inspect what Snakemake intends to execute:

```cmd
run_workflow.bat -n
```

---

## Clean reproducibility test

A simple way to test whether the repository can be reproduced from scratch is:

1. Clone the repository into a new directory.
2. Create a completely new Conda environment.
3. Copy only the required raw input files into `resources/`.
4. Do **not** copy:
   - `results/`
   - `.snakemake/`
   - cached files
   - previously fitted models
5. Run:

```cmd
run_workflow.bat -n
run_workflow.bat check
run_workflow.bat preprocess
run_workflow.bat main_model
run_workflow.bat hypotheses
run_workflow.bat requested_plots
run_workflow.bat maps
```

Finally run:

```cmd
run_workflow.bat
```

to verify that the complete default workflow is consistent.

---

## Repository structure

The most important directories are:

```text
natural_hazard_solidarity/
│
├── README.md
├── environment.yaml
├── run_workflow.bat
├── local_settings.example.bat
│
├── config/
│   └── config.yaml
│
├── docs/
│   ├── workflow.md
│   ├── data.md
│   ├── architecture.md
│   └── migration.md
│
├── resources/
├── src/
├── tests/
├── workflow/
└── results/
```

### `src/`

Contains the reusable Python implementation of the scientific analyses.

### `workflow/`

Contains the Snakemake workflow, rules, and workflow scripts.

### `config/`

Contains reproducible project configuration.

### `resources/`

Contains external/raw input files that are not generated by the workflow.

### `results/`

Contains generated datasets, fitted models, tables, and figures.

### `docs/`

Contains more detailed project documentation.

---

## Documentation

More detailed documentation is available under `docs/`:

- [`docs/workflow.md`](docs/workflow.md)  
  Workflow targets, analysis functionality, and generated outputs.

- [`docs/data.md`](docs/data.md)  
  Required input files, data sources, and expected resource structure.

- [`docs/architecture.md`](docs/architecture.md)  
  Code organization and guidance on where new functionality should be implemented.

- [`docs/migration.md`](docs/migration.md)  
  Notes on the migration from earlier analysis implementations to the current workflow.

---

## Recommended workflow for development

When modifying the analysis code:

1. Make the code change.
2. Inspect the planned Snakemake execution:

```cmd
run_workflow.bat -n
```

3. Run the relevant target, for example:

```cmd
run_workflow.bat hypotheses
```

4. Check the generated outputs.
5. Run the project tests:

```cmd
run_workflow.bat check
```

6. Before committing larger changes, verify the complete workflow:

```cmd
run_workflow.bat -n
```

---

## Reproducibility

The workflow separates:

- raw external inputs in `resources/`
- reproducible configuration in `config/`
- reusable analysis code in `src/`
- workflow orchestration in `workflow/`
- generated outputs in `results/`

This structure is intended to make the complete analysis traceable and reproducible from the original input data to the final figures and tables.