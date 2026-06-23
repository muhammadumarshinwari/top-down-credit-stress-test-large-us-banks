"""
End-to-end demo: fit the satellite model, apply an adverse scenario,
project capital, and plot the results.

Uses real FDIC call-report data for 10 major US banks (2005 Q1–2024 Q4),
merged with bundled FRED macro history. FDIC pulls are cached under
data/cache/ after the first run; the first fetch requires internet.

Run from the repo root:  python examples/run_stress_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stresskit import (
    SatelliteNPLModel, project_capital, summarize,
    load_fed_scenario, SAMPLE_CERTS, fetch_bank_financials,
)
from stresskit.models import REGRESSORS

# Actual history through 2024 Q4; Fed 2025 severely adverse scenario starts 2025 Q1.
ACTUAL_DATA_END = "2024-12-31"

print("Fetching FDIC bank financials (refreshes cache if older than target end) ...")
raw = fetch_bank_financials(
    certs=list(SAMPLE_CERTS.keys()),
    start="2005-01-01",
    end=ACTUAL_DATA_END,
    cache_dir=Path(__file__).parent.parent / "data" / "cache",
)

# convert units: FDIC reports assets, loans, equity in $000s
raw["total_assets"] = raw["total_assets"] * 1000
raw["net_loans"]    = raw["net_loans"]    * 1000
raw["equity"]       = raw["equity"]       * 1000

# align FDIC report dates to quarter-end so they merge with FRED
raw["date"] = raw["date"] + pd.offsets.QuarterEnd(0)

# load real FRED macro history — same training window as FDIC panel
macro_path = Path(__file__).parent.parent / "data" / "fred_macro_history.csv"
macro = pd.read_csv(macro_path, parse_dates=["date"])
macro = macro[
    (macro["date"] >= "2005-01-01") & (macro["date"] <= ACTUAL_DATA_END)
][[
    "date", "real_gdp_growth", "unemployment", "t3m",
    "cpi_inflation", "mortgage_rate", "house_price_index",
]].reset_index(drop=True)

# merge bank panel with macro history on date
panel = raw.merge(macro, on="date", how="inner")
panel = panel.dropna(subset=[
    "npl_ratio", "real_gdp_growth", "unemployment", "t3m",
    "cpi_inflation", "mortgage_rate", "house_price_index",
])
panel = panel.sort_values(["bank_id", "date"]).reset_index(drop=True)

names = SAMPLE_CERTS
print(f"Panel: {panel['bank_id'].nunique()} banks, {panel['date'].nunique()} quarters")

panel_path = Path(__file__).parent.parent / "data" / "regression_panel.csv"
panel_out = panel.copy()
panel_out["bank"] = panel_out["bank_id"].map(names)
panel_out = panel_out[
    ["bank_id", "bank"]
    + [c for c in panel_out.columns if c not in ("bank_id", "bank")]
]
panel_out.to_csv(panel_path, index=False)
print(f"Panel saved to {panel_path}")

# ---------------------------------------------------------------- model
model = SatelliteNPLModel().fit(panel)
print(f"Regression sample: {int(model.result.nobs)} observations (after 2-quarter lags)")
print("Satellite model coefficients:")
coef_tbl = pd.DataFrame({
    "coef": model.result.params[REGRESSORS],
    "std_err": model.result.bse[REGRESSORS],
    "p_value": model.result.pvalues[REGRESSORS],
}).round(4)
print(coef_tbl.to_string())
print(f"\nR-squared: {model.result.rsquared:.4f}  Adj R-squared: {model.result.rsquared_adj:.4f}")

bt = model.backtest(panel)
rmse = np.sqrt((bt["error"] ** 2).mean())
mae = bt["error"].abs().mean()
print(f"\nIn-sample one-step RMSE: {rmse:.3f} pp of NPL ratio")
print(f"In-sample one-step MAE:  {mae:.3f} pp of NPL ratio")

data_dir = Path(__file__).parent.parent / "data"
docs_dir = Path(__file__).parent.parent / "docs"
docs_dir.mkdir(exist_ok=True)

bt_out = bt.copy()
bt_out["bank"] = bt_out["bank_id"].map(names)
bt_out = bt_out[["bank_id", "bank", "date", "npl_ratio", "npl_pred", "error"]]
bt_path = data_dir / "backtest_results.csv"
bt_out.sort_values(["bank_id", "date"]).to_csv(bt_path, index=False)
print(f"Backtest saved to {bt_path}")

by_bank = (
    bt_out.assign(abs_error=bt_out["error"].abs())
    .groupby(["bank_id", "bank"], as_index=False)
    .agg(
        rmse=("error", lambda s: np.sqrt((s**2).mean())),
        mae=("abs_error", "mean"),
    )
    .sort_values("rmse")
)
print("\nBacktest by bank (RMSE / MAE, pp of NPL ratio):")
print(by_bank.round(4).to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
for _, g in bt_out.sort_values("date").groupby("bank_id"):
    axes[0].plot(g["date"], g["npl_ratio"], alpha=0.35, lw=1)
    axes[0].plot(g["date"], g["npl_pred"], alpha=0.35, lw=1, ls="--")
axes[0].set_title("Backtest: actual NPL (solid) vs one-step predicted (dashed)")
axes[0].set_ylabel("NPL ratio (%)")

err_ts = bt_out.groupby("date", as_index=False)["error"].mean()
axes[1].bar(err_ts["date"], err_ts["error"], width=60, color="steelblue", alpha=0.7)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Average prediction error by quarter (actual - predicted)")
axes[1].set_ylabel("Error (pp)")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
bt_plot = docs_dir / "backtest_results.png"
fig.savefig(bt_plot, dpi=120)
plt.close(fig)
print(f"Backtest chart saved to {bt_plot}")

# ---------------------------------------------------------------- stress
scenario = load_fed_scenario(
    Path(__file__).parent.parent / "data" / "2025-Table_3A_Supervisory_Severely_Adverse_Domestic.csv"
)
last = panel.sort_values("date").groupby("bank_id").last().reset_index()

npl_paths = model.project(last[["bank_id", "npl_ratio"]], scenario)
# PPNR is compressed under stress (margin compression + lower fee income)
projection = project_capital(
    last[["bank_id", "npl_ratio", "net_loans", "total_assets", "equity"]],
    npl_paths,
    assumptions={"ppnr_roa_q": 0.0010, "lgd": 0.45},
)

results = summarize(projection, names)
print("\nStress test results (2025 Fed supervisory severely adverse scenario):")
print(results.to_string(index=False))

# ---------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for bank_id, g in projection.groupby("bank_id"):
    axes[0].plot(g["date"], g["npl_ratio"], alpha=0.7)
axes[0].set_title("Projected NPL ratios under adverse scenario")
axes[0].set_ylabel("NPL ratio (%)")

for bank_id, g in projection.groupby("bank_id"):
    axes[1].plot(g["date"], g["capital_ratio"] * 100, alpha=0.7)
axes[1].axhline(5.0, color="red", ls="--", lw=1, label="5% minimum")
axes[1].set_title("Projected capital ratios")
axes[1].set_ylabel("Equity / assets (%)")
axes[1].legend()

for ax in axes:
    ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
out = docs_dir / "stress_results.png"
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"\nChart saved to {out}")

