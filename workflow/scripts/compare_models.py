"""LOO for identical observations only; not a ranking of different respondent subsamples."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from natural_hazard_solidarity.posterior import validate_model_comparison


def main(s):
    import arviz as az
    models = {name: az.from_netcdf(path) for name, path in zip(s.params.runs, s.input.models)}
    validate_model_comparison(models)
    loo = {name: az.loo(idata, var_name="choice", pointwise=True) for name, idata in models.items()}
    table = az.compare(loo, ic="loo", scale="log", method="stacking")
    Path(s.output.table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(s.output.table, index_label="run")
    pd.concat([pd.DataFrame({"run": name, "task_id": models[name].observed_data.task.values,
                             "pareto_k": result.pareto_k.values.reshape(-1),
                             "good_k": result.good_k}) for name, result in loo.items()], ignore_index=True).to_csv(s.output.pareto, index=False)


if __name__ == "__main__":
    main(globals()["snakemake"])
