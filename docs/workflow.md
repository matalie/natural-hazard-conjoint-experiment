# Workflow

The analysis is managed with Snakemake and started in the commandline in your project folder through:

```cmd
run_workflow.bat <target>
```

Running the command without a target executes the complete default workflow:

```cmd
run_workflow.bat
```

The default workflow covers the main analysis pipeline from raw input data to
the figures and tables used in the analysis. It includes:

- preprocessing and merging of the two survey waves
- construction of the final analysis sample
- preparation and encoding of the conjoint experiment
- estimation of the main Bayesian hierarchical conjoint model
- descriptive and Likert-scale analyses
- exploratory factor analysis and latent-construct summaries
- respondent-level posterior analyses
- hypothesis analyses (H1-H4)
- model and sampling diagnostics
- survey-based and posterior geographic maps

A dry run of any workflow can be used to inspect planned jobs adding `-n`:

```cmd
run_workflow.bat -n
```

### Main targets

| Target | Purpose | Main outputs |
|---|---|---|
|`conjoint_data`| Preprocess survey data + create conjoint dataset | `results/data/conjoint_(effect|dummy).parquet` |
| `main_model` | Fit the main Bayesian conjoint model | `results/models/main.nc` |
| `hypotheses` | Generate H1-H4 analyses | `results/figures/hypotheses/`, `results/tables/hypotheses/` |
| `requested_plots` | Generate descriptive and exploratory figures | `results/figures/` |
| `diagnostics` | Export posterior and sampling diagnostics | `results/tables/diagnostics/` |
| `maps` | Generate survey and posterior maps | `results/figures/maps/`, `results/tables/maps/` |
| `robustness_plots` | Run robustness analyses | `results/figures/robustness/`, `results/tables/robustness/` |
| `model_comparison` | Compare configured Bayesian models | `results/tables/comparisons/` |

Preprocessing steps are automatically executed whenever required by a
downstream target.

### Main dependency structure

```text
raw survey data
       ↓
preprocessing
       ↓
analysis sample
       ↓
conjoint data
       ↓
main Bayesian model
       ↓
hypotheses / diagnostics / maps / robustness
```

Snakemake automatically reruns outputs that are missing or affected by
updated inputs or source files.

### Output structure

```text
results/
├── data/
├── models/
├── figures/
│   ├── hypotheses/
│   ├── maps/
│   └── robustness/
└── tables/
    ├── diagnostics/
    ├── hypotheses/
    ├── maps/
    └── robustness/
```

### Workflow graph

The Snakemake DAG can be generated with:

```cmd
run_workflow.bat --dag
```

With Graphviz installed:

```cmd
run_workflow.bat --rulegraph | dot -Tsvg > workflow_rulegraph.svg
```