"""Out-of-sample forecast: est through 2022 Q4, forecast 2023 Q1–2024 Q4."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stresskit import SAMPLE_CERTS, SatelliteNPLModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

panel = pd.read_csv(DATA_DIR / "regression_panel.csv", parse_dates=["date"])
panel = panel.sort_values(["bank_id", "date"])

est_end = pd.Timestamp("2022-12-31")
fcast_start = pd.Timestamp("2023-03-31")
fcast_end = pd.Timestamp("2024-12-31")

est = panel[panel["date"] <= est_end].copy()
fcast = panel[(panel["date"] >= fcast_start) & (panel["date"] <= fcast_end)].copy()

model = SatelliteNPLModel().fit(est)

# Build lags on full panel so macro/NPL history before 2023 is available
full_bt = model.backtest(panel)
bt = full_bt[(full_bt["date"] >= fcast_start) & (full_bt["date"] <= fcast_end)].copy()
bt["bank"] = bt["bank_id"].map(SAMPLE_CERTS)
bt["abs_error"] = bt["error"].abs()

rmse = float(np.sqrt((bt["error"] ** 2).mean()))
mae = float(bt["abs_error"].mean())
bias = float(bt["error"].mean())

by_bank = (
    bt.groupby(["bank_id", "bank"], as_index=False)
    .agg(
        n_obs=("error", "count"),
        rmse=("error", lambda s: np.sqrt((s**2).mean())),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    )
    .sort_values("rmse")
)

print("=" * 72)
print("OUT-OF-SAMPLE FORECAST")
print("=" * 72)
print(f"Estimation:   {est['date'].min().date()} to {est['date'].max().date()}  ({len(est)} rows)")
print(f"Forecasting:  {fcast_start.date()} to {fcast_end.date()}  (requested)")
if len(fcast):
    print(
        f"Data available: {fcast['date'].min().date()} to {fcast['date'].max().date()}  "
        f"({len(fcast)} rows, {len(bt)} used after lags)"
    )
else:
    print("Data available: none")
print()
print("FORECAST SAMPLE PERFORMANCE (one-step ahead)")
print("-" * 72)
print(f"RMSE: {rmse:.4f} pp")
print(f"MAE:  {mae:.4f} pp")
print(f"Bias: {bias:.4f} pp")
print()
print("BY BANK")
print("-" * 72)
print(by_bank.round(4).to_string(index=False))
print()
print("QUARTER BY QUARTER (panel average)")
print("-" * 72)
qavg = (
    bt.groupby("date", as_index=False)
    .agg(actual=("npl_ratio", "mean"), predicted=("npl_pred", "mean"), error=("error", "mean"))
)
print(qavg.round(4).to_string(index=False))

bt_out = bt[["bank_id", "bank", "date", "npl_ratio", "npl_pred", "error"]].sort_values(
    ["bank_id", "date"]
)
bt_out.to_csv(DATA_DIR / "forecast_sample_results.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
for _, g in bt_out.groupby("bank_id"):
    axes[0].plot(g["date"], g["npl_ratio"], alpha=0.5, lw=1.2)
    axes[0].plot(g["date"], g["npl_pred"], alpha=0.5, lw=1.2, ls="--")
axes[0].set_title("Forecast sample: actual (solid) vs predicted (dashed)")
axes[0].set_ylabel("NPL ratio (%)")
err_ts = bt.groupby("date")["error"].mean().reset_index()
axes[1].bar(err_ts["date"], err_ts["error"], width=50, color="steelblue", alpha=0.7)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("Error (pp)")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(DOCS_DIR / "forecast_sample_results.png", dpi=120)
plt.close()

print()
print(f"Saved: {DATA_DIR / 'forecast_sample_results.csv'}")
print(f"Chart: {DOCS_DIR / 'forecast_sample_results.png'}")
