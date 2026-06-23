"""Model validation for satellite NPL model (estimation through 2022 Q4)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stresskit import SatelliteNPLModel, validate_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
EST_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-03-31")
OOS_END = pd.Timestamp("2024-12-31")

panel = pd.read_csv(DATA_DIR / "regression_panel.csv", parse_dates=["date"])
est = panel[panel["date"] <= EST_END].copy()

model = SatelliteNPLModel().fit(est)
report = validate_model(
    model,
    panel,
    est_end=EST_END,
    oos_start=OOS_START,
    oos_end=OOS_END,
)

report.print_summary()
print()
print("VIF (multicollinearity; >5-10 suggests concern)")
print(report.vif.to_string(index=False))
print()
print("REGRESSOR CORRELATION")
print(report.regressor_correlation.to_string())
print()
print("RESIDUAL NORMALITY")
print(report.normality.to_string(index=False))
print()
print("LJUNG-BOX (serial correlation in residuals)")
print(report.ljung_box.round(4).to_string())
print()
print("HETEROSKEDASTICITY")
print(report.heteroskedasticity.to_string(index=False))
print()
print("IN-SAMPLE RMSE BY PERIOD")
print(report.backtest_by_period.to_string())
print()
print("IN-SAMPLE RMSE BY BANK")
print(report.backtest_by_bank.to_string(index=False))
if report.out_of_sample_by_bank is not None:
    print()
    print("OUT-OF-SAMPLE BY BANK")
    print(report.out_of_sample_by_bank.to_string(index=False))

report.save(DATA_DIR)
DOCS_DIR.mkdir(exist_ok=True)
report.plot(DOCS_DIR / "model_diagnostics.png")

print()
print(f"Chart: {DOCS_DIR / 'model_diagnostics.png'}")
print(f"CSV files saved to: {DATA_DIR}")
