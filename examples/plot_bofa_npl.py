import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

panel = pd.read_csv(
    Path(__file__).resolve().parent.parent / "data" / "regression_panel.csv",
    parse_dates=["date"],
)
bofa = panel[panel["bank_id"] == 3510].sort_values("date")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(bofa["date"], bofa["npl_ratio"], color="#012169", lw=2, marker="o", markersize=3)
ax.set_title("Bank of America NA — NPL Ratio (2005 Q1–2023 Q4)")
ax.set_xlabel("Quarter")
ax.set_ylabel("NPL ratio (%)")
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "docs" / "bofa_npl_ratio.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
plt.close()

print(f"Saved to {out}")
print(f"Observations: {len(bofa)}")
print(f"Min: {bofa['npl_ratio'].min():.2f}%")
print(f"Max: {bofa['npl_ratio'].max():.2f}%")
print(f"Latest: {bofa['npl_ratio'].iloc[-1]:.2f}%")
