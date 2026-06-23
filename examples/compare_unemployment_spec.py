"""Compare unemployment lag-only vs lag + contemporaneous specifications."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stresskit.models import REGRESSORS, REGRESSORS_LAG_ONLY, SatelliteNPLModel

panel = pd.read_csv(
    Path(__file__).resolve().parent.parent / "data" / "regression_panel.csv",
    parse_dates=["date"],
)


def fit_summary(regs: list[str], label: str) -> dict:
    model = SatelliteNPLModel(regressors=regs).fit(panel)
    bt = model.backtest(panel)
    rmse = float(np.sqrt((bt["error"] ** 2).mean()))
    mae = float(bt["error"].abs().mean())
    coef = model.result.params[regs]
    pval = model.result.pvalues[regs]
    return {
        "label": label,
        "n_obs": int(model.result.nobs),
        "r2": model.result.rsquared,
        "adj_r2": model.result.rsquared_adj,
        "rmse": rmse,
        "mae": mae,
        "coef": coef,
        "pval": pval,
    }


lag_only = fit_summary(REGRESSORS_LAG_ONLY, "Lag only (no contemporaneous unemployment)")
with_contemp = fit_summary(REGRESSORS, "Lag + contemporaneous unemployment")

print("=" * 88)
print("UNEMPLOYMENT: LAG-ONLY vs LAG + CONTEMPORANEOUS")
print("=" * 88)
print()

fit_tbl = pd.DataFrame([
    {
        "spec": lag_only["label"],
        "n_obs": lag_only["n_obs"],
        "r2": round(lag_only["r2"], 4),
        "adj_r2": round(lag_only["adj_r2"], 4),
        "rmse_pp": round(lag_only["rmse"], 4),
        "mae_pp": round(lag_only["mae"], 4),
    },
    {
        "spec": with_contemp["label"],
        "n_obs": with_contemp["n_obs"],
        "r2": round(with_contemp["r2"], 4),
        "adj_r2": round(with_contemp["adj_r2"], 4),
        "rmse_pp": round(with_contemp["rmse"], 4),
        "mae_pp": round(with_contemp["mae"], 4),
    },
])
print("MODEL FIT")
print("-" * 88)
print(fit_tbl.to_string(index=False))
print()

for res in (lag_only, with_contemp):
    print("-" * 88)
    print(res["label"].upper())
    print("-" * 88)
    tbl = pd.DataFrame({"coef": res["coef"], "p_value": res["pval"]}).round(4)
    print(tbl.to_string())
    print()

out = Path(__file__).resolve().parent.parent / "data" / "unemployment_spec_comparison.csv"
rows = []
for res in (lag_only, with_contemp):
    for var in res["coef"].index:
        rows.append({
            "spec": res["label"],
            "variable": var,
            "coef": res["coef"][var],
            "p_value": res["pval"][var],
            "r2": res["r2"],
            "rmse_pp": res["rmse"],
        })
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved to {out}")
