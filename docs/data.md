# Input data

The raw input data are not included in this repository.

All external input files must be placed in:

```text
resources/
```

## Required files

```text
resources/
├── Survey_Adaptation Natural Hazards_First Wave_raw data.csv
├── Survey_Adaptation Natural Hazards_Second Wave_raw data.csv
├── id_list.csv
├── swissBOUNDARIES3D_1_5_LV95_LN02.gpkg
└── AMTOVZ_CSV_WGS84.csv
```

| File | Purpose |
|---|---|
| First-wave survey CSV | Raw survey data from wave 1 |
| Second-wave survey CSV | Raw survey data from wave 2 |
| `id_list.csv` | Links respondents across survey waves |
| `swissBOUNDARIES3D_1_5_LV95_LN02.gpkg` | Swiss administrative boundaries used for maps |
| `AMTOVZ_CSV_WGS84.csv` | Municipality/postcode geographic lookup |

The workflow reads these files automatically according to the paths defined
in `config/config.yaml`.

Generated and cleaned datasets are written to:

```text
results/data/
```

Raw data in `resources/` should not be modified by the workflow.