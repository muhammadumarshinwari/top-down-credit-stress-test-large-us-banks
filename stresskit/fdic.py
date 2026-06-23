"""
Pull bank-level financial data from the FDIC BankFind Suite API.

The FDIC publishes quarterly financials for every FDIC-insured institution.
No API key required. Docs: https://banks.data.fdic.gov/docs/

We pull a small set of fields needed for a simplified stress test:

    ASSET   Total assets ($000)
    DEP     Total deposits ($000)
    LNLSNET Net loans and leases ($000)
    NPTLA   Noncurrent loans / total loans (%)  -> our NPL ratio proxy
    NIMY    Net interest margin (%)
    ROA     Return on assets (%)
    RBCT1J  Tier 1 leverage / CET1-related capital measure
    RBC1RWAJ Tier 1 risk-based capital ratio (%)
    EQ      Total equity capital ($000)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://banks.data.fdic.gov/api/financials"

DEFAULT_FIELDS = [
    "CERT", "REPDTE", "ASSET", "DEP", "LNLSNET",
    "NCLNLS", "NIMY", "ROA", "RBC1RWAJ", "EQ",
]


def fetch_bank_financials(
    certs: list[int],
    start: str = "2010-03-31",
    end: str = "2025-12-31",
    fields: list[str] | None = None,
    cache_dir: str | Path = "data/cache",
    pause: float = 0.25,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Download quarterly financials for a list of FDIC certificate numbers.

    Results are cached to CSV per bank so re-runs don't hit the API.
    Re-fetches automatically when the cache ends before ``end`` or when
    ``refresh=True``. Returns a long panel: one row per bank-quarter.
    """
    fields = fields or DEFAULT_FIELDS
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    end_ts = pd.to_datetime(end)
    start_api = pd.to_datetime(start).strftime("%Y%m%d")
    end_api = end_ts.strftime("%Y%m%d")

    frames = []
    for cert in certs:
        cached = cache_dir / f"fdic_{cert}.csv"
        use_cache = cached.exists() and not refresh
        if use_cache:
            cached_df = pd.read_csv(cached)
            cached_df["REPDTE"] = pd.to_datetime(cached_df["REPDTE"], format="%Y%m%d")
            if cached_df["REPDTE"].max() < end_ts:
                use_cache = False

        if use_cache:
            frames.append(pd.read_csv(cached))
            continue

        params = {
            "filters": f"CERT:{cert} AND REPDTE:[{start_api} TO {end_api}]",
            "fields": ",".join(fields),
            "sort_by": "REPDTE",
            "sort_order": "ASC",
            "limit": 10_000,
            "format": "json",
        }
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = [d["data"] for d in resp.json().get("data", [])]
        df = pd.DataFrame(rows)
        df.to_csv(cached, index=False)
        frames.append(df)
        time.sleep(pause)  # be polite to the public API

    panel = pd.concat(frames, ignore_index=True)
    panel["REPDTE"] = pd.to_datetime(panel["REPDTE"], format="%Y%m%d")
    panel = panel.rename(
        columns={
            "CERT": "bank_id",
            "REPDTE": "date",
            "ASSET": "total_assets",
            "DEP": "deposits",
            "LNLSNET": "net_loans",
            "NCLNLS": "noncurrent_loans",
            "NIMY": "nim",
            "ROA": "roa",
            "RBC1RWAJ": "tier1_ratio",
            "EQ": "equity",
        }
    )
    # compute NPL ratio: noncurrent loans / net loans (as percentage)
    panel["npl_ratio"] = (panel["noncurrent_loans"] / panel["net_loans"]) * 100
    return panel.sort_values(["bank_id", "date"]).reset_index(drop=True)


# A starter sample: 10 large/regional US banks by FDIC certificate number.
# Look up any institution at https://banks.data.fdic.gov/bankfind-suite/bankfind
SAMPLE_CERTS = {
    628: "JPMorgan Chase Bank NA",
    3510: "Bank of America NA",
    3511: "Wells Fargo Bank NA",
    7213: "Citibank NA",
    6548: "U.S. Bank NA",
    6384: "PNC Bank NA",
    9846: "Truist Bank",
    5649: "Fifth Third Bank NA",
    6672: "KeyBank NA",
    18409: "Regions Bank",
}
