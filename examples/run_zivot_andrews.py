"""Run Zivot-Andrews unit root tests (allowing one structural break)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arch.unitroot import ZivotAndrews

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stresskit.models import _add_gdp_lags

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALPHA = 0.05


def za_test(series: pd.Series, dates: pd.Series, name: str, group: str) -> dict:
    y = series.dropna().astype(float).values
    d = dates.loc[series.dropna().index]
    za = ZivotAndrews(y, lags=1, trend="c")

    nobs = len(y)
    trimcnt = int(nobs * za._trim)
    start = trimcnt + 1
    end = nobs - trimcnt
    stats = za._all_stats[start : end + 1]
    bps = np.arange(start, end + 1)
    bp = int(bps[np.argmin(stats)])
    break_date = d.iloc[bp - 1] if bp - 1 < len(d) else pd.NaT

    return {
        "variable": name,
        "group": group,
        "n_obs": nobs,
        "test_stat": za.stat,
        "p_value": za.pvalue,
        "lags": za.lags,
        "break_period": bp,
        "break_date": break_date.date() if pd.notna(break_date) else None,
        "crit_1pct": za.critical_values["1%"],
        "crit_5pct": za.critical_values["5%"],
        "crit_10pct": za.critical_values["10%"],
        "stationary_5pct": za.pvalue < ALPHA,
    }


panel = pd.read_csv(DATA_DIR / "regression_panel.csv", parse_dates=["date"])
panel = _add_gdp_lags(panel)
panel["npl_lag"] = panel.groupby("bank_id")["npl_ratio"].shift(1)

rows = []

macro = (
    panel[["date", "real_gdp_growth", "real_gdp_growth_lag1",
           "real_gdp_growth_lag2", "unemployment", "t3m"]]
    .drop_duplicates("date")
    .sort_values("date")
    .reset_index(drop=True)
)
for col in macro.columns:
    if col == "date":
        continue
    rows.append(za_test(macro[col], macro["date"], col, "macro (pooled dates)"))

for bank_id, g in panel.groupby("bank_id"):
    bank = g["bank"].iloc[0] if "bank" in g.columns else str(bank_id)
    g = g.sort_values("date").reset_index(drop=True)
    rows.append(za_test(g["npl_ratio"], g["date"], "npl_ratio", bank))
    rows.append(za_test(g["npl_lag"], g["date"], "npl_lag", bank))

results = pd.DataFrame(rows).sort_values(["group", "variable"]).reset_index(drop=True)
out = DATA_DIR / "zivot_andrews_tests.csv"
results.to_csv(out, index=False, float_format="%.6f")

bofa = results[
    (results["group"] == "Bank of America NA") & (results["variable"] == "npl_ratio")
].iloc[0]
bofa_summary = pd.DataFrame([
    {"field": "Bank", "value": "Bank of America NA"},
    {"field": "Variable", "value": "npl_ratio"},
    {"field": "Test", "value": "Zivot-Andrews (constant, 1 structural break)"},
    {"field": "Observations", "value": int(bofa["n_obs"])},
    {"field": "ZA test statistic", "value": round(bofa["test_stat"], 4)},
    {"field": "p-value", "value": round(bofa["p_value"], 4)},
    {"field": "Lags", "value": int(bofa["lags"])},
    {"field": "Estimated break date", "value": bofa["break_date"]},
    {"field": "Critical value (1%)", "value": round(bofa["crit_1pct"], 4)},
    {"field": "Critical value (5%)", "value": round(bofa["crit_5pct"], 4)},
    {"field": "Critical value (10%)", "value": round(bofa["crit_10pct"], 4)},
    {"field": "H0", "value": "Unit root with a single structural break"},
    {"field": "Reject H0 at 5%?", "value": "Yes" if bofa["stationary_5pct"] else "No"},
    {"field": "Conclusion at 5%", "value": "Trend-break stationary" if bofa["stationary_5pct"] else "Unit root (even allowing break)"},
])
bofa_out = DATA_DIR / "bofa_npl_zivot_andrews_summary.csv"
bofa_summary.to_csv(bofa_out, index=False)

print("=" * 92)
print("ZIVOT-ANDREWS UNIT ROOT TESTS (one endogenous structural break)")
print("H0: unit root with a break. Reject H0 at 5% => trend-break stationary.")
print("=" * 92)
print()

macro_tbl = results[results["group"] == "macro (pooled dates)"]
print("MACRO VARIABLES")
print("-" * 92)
print(
    macro_tbl[
        ["variable", "n_obs", "test_stat", "p_value", "break_date", "stationary_5pct"]
    ].round(4).to_string(index=False)
)
print()

bank_tbl = results[results["group"] != "macro (pooled dates)"]
print("BANK-LEVEL npl_ratio (p-values and estimated break dates)")
print("-" * 92)
show = bank_tbl[bank_tbl["variable"] == "npl_ratio"].sort_values("p_value")
print(
    show[["group", "test_stat", "p_value", "break_date", "stationary_5pct"]]
    .round(4).to_string(index=False)
)
print()
print("BofA summary table:")
print(bofa_summary.to_string(index=False))
print()
print(f"Full results: {out}")
print(f"BofA summary: {bofa_out}")
