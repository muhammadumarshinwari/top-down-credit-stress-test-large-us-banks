"""
Macro scenario handling.

The Federal Reserve publishes its supervisory stress test scenarios
(baseline and severely adverse) as CSVs each February:
https://www.federalreserve.gov/supervisionreg/dfast-archive.htm

Download the "Historic and projected variables" CSV, then load it with
`load_fed_scenario`. A clearly-labeled ILLUSTRATIVE scenario is bundled in
data/illustrative_scenario.csv so the toolkit runs end-to-end out of the box —
its numbers are NOT the Fed's published scenario.

A scenario here is a quarterly DataFrame with columns:

    date, real_gdp_growth, unemployment, t3m, t10y, equity_return
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "date", "real_gdp_growth", "unemployment", "t3m", "t10y", "equity_return",
]

# Map of Fed CSV column names -> toolkit names (Fed names vary slightly by year;
# adjust after inspecting the file you download).
FED_COLUMN_MAP = {
    "Date": "date",
    "Real GDP growth": "real_gdp_growth",
    "Unemployment rate": "unemployment",
    "CPI inflation rate": "cpi_inflation",
    "3-month Treasury rate": "t3m",
    "10-year Treasury yield": "t10y",
    "Mortgage rate": "mortgage_rate",
    "House Price Index (Level)": "house_price_index",
    "Dow Jones Total Stock Market Index (Level)": "equity_level",
}

MODEL_MACRO_COLUMNS = [
    "real_gdp_growth",
    "unemployment",
    "t3m",
    "cpi_inflation",
    "mortgage_rate",
    "house_price_index",
]


def load_fed_scenario(csv_path: str | Path) -> pd.DataFrame:
    """Load a Fed supervisory scenario CSV and normalize column names."""
    raw = pd.read_csv(csv_path)
    cols = {c: FED_COLUMN_MAP[c] for c in raw.columns if c in FED_COLUMN_MAP}
    df = raw.rename(columns=cols)
    # Fed CSV dates look like "2025 Q1" — remove the space so pandas parses them
    df["date"] = pd.PeriodIndex(
        df["date"].str.replace(r"\s+", "", regex=True), freq="Q"
    ).to_timestamp(how="end")
    if "equity_level" in df.columns:
        df["equity_return"] = df["equity_level"].pct_change().fillna(0) * 100
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Scenario file is missing {missing}. "
            "Inspect the Fed CSV and update FED_COLUMN_MAP."
        )
    missing_model = [c for c in MODEL_MACRO_COLUMNS if c not in df.columns]
    if missing_model:
        raise ValueError(
            f"Scenario file is missing model macro columns {missing_model}. "
            "Inspect the Fed CSV and update FED_COLUMN_MAP."
        )
    cols = list(dict.fromkeys(REQUIRED_COLUMNS + MODEL_MACRO_COLUMNS))
    return df[cols]


def load_illustrative_scenario(
    path: str | Path = Path(__file__).parent.parent / "data" / "illustrative_scenario.csv",
) -> pd.DataFrame:
    """Bundled 9-quarter illustrative adverse scenario (NOT a Fed scenario)."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df[REQUIRED_COLUMNS]
