"""Phillips-Perron summary table for Bank of America NPL ratio."""
import sys
from pathlib import Path

import pandas as pd
from arch.unitroot import PhillipsPerron

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
panel = pd.read_csv(DATA_DIR / "regression_panel.csv", parse_dates=["date"])
bofa = panel[panel["bank_id"] == 3510].sort_values("date")
y = bofa["npl_ratio"].astype(float)

pp = PhillipsPerron(y, lags=None)
alpha = 0.05
period = f"{bofa['date'].min().date()} to {bofa['date'].max().date()}"

summary = pd.DataFrame([
    {"field": "Bank", "value": "Bank of America NA"},
    {"field": "FDIC cert (bank_id)", "value": 3510},
    {"field": "Variable", "value": "npl_ratio"},
    {"field": "Sample period", "value": period},
    {"field": "Observations", "value": len(y)},
    {"field": "PP test statistic", "value": round(pp.stat, 4)},
    {"field": "p-value", "value": round(pp.pvalue, 4)},
    {"field": "Lags (automatic)", "value": pp.lags},
    {"field": "Critical value (1%)", "value": round(pp.critical_values["1%"], 4)},
    {"field": "Critical value (5%)", "value": round(pp.critical_values["5%"], 4)},
    {"field": "Critical value (10%)", "value": round(pp.critical_values["10%"], 4)},
    {"field": "H0", "value": "Unit root (non-stationary)"},
    {"field": "Reject H0 at 5%?", "value": "Yes" if pp.pvalue < alpha else "No"},
    {"field": "Conclusion at 5%", "value": "Stationary" if pp.pvalue < alpha else "Non-stationary"},
    {"field": "Series mean (%)", "value": round(y.mean(), 4)},
    {"field": "Series min (%)", "value": round(y.min(), 4)},
    {"field": "Series max (%)", "value": round(y.max(), 4)},
])

out = DATA_DIR / "bofa_npl_pp_test_summary.csv"
summary.to_csv(out, index=False)
print(summary.to_string(index=False))
print(f"\nSaved to {out}")
