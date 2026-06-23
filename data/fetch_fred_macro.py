"""
Downloads all macro variables matching the Fed DFAST severely adverse
domestic scenario from FRED (no API key required) and saves a clean
quarterly panel to data/fred_macro_history.csv.

Run from the repo root:
    python data/fetch_fred_macro.py
"""

from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
START = "2000-01-01"

# FRED series ID → (output column name, frequency, transform)
# frequency: "Q" = already quarterly, "M" = monthly
# transform: "level"           = use as-is (quarterly resample: last value)
#            "avg"             = quarterly average of monthly
#            "pct_change_yoy"  = quarterly level then year-over-year % change
SERIES = {
    "A191RL1Q225SBEA": ("real_gdp_growth",                  "Q", "level"),
    "A191RP1Q027SBEA": ("nominal_gdp_growth",               "Q", "level"),
    "DPIC96":          ("real_disposable_income_growth",    "M", "pct_change_yoy"),
    "DSPI":            ("nominal_disposable_income_growth", "M", "pct_change_yoy"),
    "UNRATE":          ("unemployment",                     "M", "avg"),
    "CPIAUCSL":        ("cpi_inflation",                    "M", "pct_change_yoy"),
    "TB3MS":           ("t3m",                              "M", "avg"),
    "GS5":             ("t5y",                              "M", "avg"),
    "GS10":            ("t10y",                             "M", "avg"),
    "DBAA":            ("bbb_corporate_yield",              "M", "avg"),  # Moody's Baa, back to 1962
    "MORTGAGE30US":    ("mortgage_rate",                    "M", "avg"),
    "DPRIME":          ("prime_rate",                       "M", "avg"),
    "SP500":           ("dow_jones_index",                  "M", "avg"),  # S&P 500 as equity proxy
    "USSTHPI":         ("house_price_index",                "Q", "level"),
    "COMREPUSQ159N":   ("cre_price_index",                  "Q", "level"),
    "VIXCLS":          ("vix",                              "M", "avg"),
}

# These must be non-null — all other columns can have NaN for early periods
CORE_COLUMNS = ["real_gdp_growth", "unemployment", "t3m"]


def fetch(series_id: str, col_name: str) -> pd.Series | None:
    url = FRED_BASE + series_id
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45)
            if r.status_code == 404:
                print(f"  SKIP {series_id} — not found on FRED")
                return None
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
            df = df.rename(columns={"observation_date": "date"})
            df = df[df["date"] >= START].copy()
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            return df.set_index("date")[series_id].dropna()
        except requests.exceptions.Timeout:
            if attempt < 2:
                print(f"  Timeout on {series_id}, retrying ({attempt + 2}/3) ...")
                time.sleep(5)
            else:
                print(f"  SKIP {series_id} — timed out after 3 attempts")
                return None
        except Exception as e:
            print(f"  SKIP {series_id} — {e}")
            return None


print("Fetching macro data from FRED ...\n")

quarterly_series = {}
for series_id, (col_name, freq, transform) in SERIES.items():
    print(f"  Downloading {series_id} ({col_name}) ...")
    s = fetch(series_id, col_name)
    if s is None:
        continue

    if freq == "M":
        if transform == "pct_change_yoy":
            s = s.resample("QE").last().pct_change(4) * 100
        else:
            s = s.resample("QE").mean()
    else:
        s = s.resample("QE").last()

    quarterly_series[col_name] = s
    time.sleep(1)  # be polite to FRED

macro = pd.DataFrame(quarterly_series)

# only require core columns to be non-null
macro = macro.dropna(subset=CORE_COLUMNS).reset_index()
macro = macro.rename(columns={"date": "date"})
if macro.columns[0] != "date":
    macro = macro.rename(columns={macro.columns[0]: "date"})

out = Path(__file__).parent / "fred_macro_history.csv"
macro.to_csv(out, index=False)

print(f"\nSaved {len(macro)} quarters ({macro['date'].min().date()} to {macro['date'].max().date()})")
print(f"Columns ({len(macro.columns)}): {list(macro.columns)}")
print(f"\nNon-null counts per column:")
print(macro.notna().sum().to_string())
