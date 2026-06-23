"""Run Phillips-Perron unit root tests on regression variables."""
import sys
from pathlib import Path

import pandas as pd
from arch.unitroot import PhillipsPerron

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stresskit.models import _add_gdp_lags

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
panel = pd.read_csv(DATA_DIR / "regression_panel.csv", parse_dates=["date"])
panel = _add_gdp_lags(panel)
panel["npl_lag"] = panel.groupby("bank_id")["npl_ratio"].shift(1)

ALPHA = 0.05


def pp_test(series: pd.Series, name: str, group: str) -> dict:
    y = series.dropna().astype(float)
    pp = PhillipsPerron(y, lags=None)
    return {
        "variable": name,
        "group": group,
        "n_obs": len(y),
        "test_stat": pp.stat,
        "p_value": pp.pvalue,
        "lags": pp.lags,
        "crit_1pct": pp.critical_values["1%"],
        "crit_5pct": pp.critical_values["5%"],
        "crit_10pct": pp.critical_values["10%"],
        "stationary_5pct": pp.pvalue < ALPHA,
    }


rows = []

macro = (
    panel[["date", "real_gdp_growth", "real_gdp_growth_lag1",
           "real_gdp_growth_lag2", "unemployment", "t3m"]]
    .drop_duplicates("date")
    .sort_values("date")
)
for col in macro.columns:
    if col == "date":
        continue
    rows.append(pp_test(macro[col], col, "macro (pooled dates)"))

for bank_id, g in panel.groupby("bank_id"):
    bank = g["bank"].iloc[0] if "bank" in g.columns else str(bank_id)
    rows.append(pp_test(g["npl_ratio"], "npl_ratio", bank))
    rows.append(pp_test(g["npl_lag"], "npl_lag", bank))

results = pd.DataFrame(rows)
results = results.sort_values(["group", "variable"]).reset_index(drop=True)

out = DATA_DIR / "phillips_perron_tests.csv"
results.to_csv(out, index=False, float_format="%.6f")

print("=" * 88)
print("PHILLIPS-PERRON UNIT ROOT TESTS")
print("H0: series has a unit root (non-stationary). Reject H0 at 5% => stationary.")
print("=" * 88)
print()

macro_tbl = results[results["group"] == "macro (pooled dates)"].copy()
print("MACRO VARIABLES (76 quarterly observations)")
print("-" * 88)
print(
    macro_tbl[
        ["variable", "n_obs", "test_stat", "p_value", "crit_5pct", "stationary_5pct"]
    ].round(4).to_string(index=False)
)
print()

bank_tbl = results[results["group"] != "macro (pooled dates)"].copy()
summary = (
    bank_tbl.groupby("variable")["stationary_5pct"]
    .agg(stationary_banks="sum", non_stationary_banks=lambda s: (~s).sum())
    .astype(int)
)
print("BANK-LEVEL SERIES (10 banks, 76 obs each)")
print("-" * 88)
print("Count of banks classified as stationary at 5%:")
print(summary.to_string())
print()

print("DETAIL BY BANK (p-values)")
print("-" * 88)
show = bank_tbl.pivot(index="group", columns="variable", values="p_value").round(4)
print(show[["npl_ratio", "npl_lag"]].to_string())
print()
print(f"Full results saved to {out}")
