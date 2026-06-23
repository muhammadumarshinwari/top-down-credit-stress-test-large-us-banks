"""
Capital projection engine.

Translates projected NPL paths into credit losses and rolls bank capital
forward quarter by quarter — a simplified version of the dynamic capital
projection used in supervisory stress tests.

Per quarter, for each bank:

    new_npls      = max(d npl_ratio, 0) * net_loans
    credit_loss   = new_npls * LGD
    pre_prov_inc  = ppnr_roa_q * total_assets        (simple PPNR proxy)
    net_income    = pre_prov_inc - credit_loss
    equity        = equity + net_income              (no distributions)
    capital_ratio = equity / total_assets            (leverage-style ratio)

Assumptions are deliberately visible and adjustable. RWA-based ratios,
NII modeling, and AFS mark-to-market are natural extensions (see README).
"""

from __future__ import annotations

import pandas as pd

DEFAULTS = {
    "lgd": 0.45,            # loss given default on newly nonperforming loans
    "ppnr_roa_q": 0.0030,   # quarterly pre-provision return on assets (0.30%)
    "min_ratio": 0.05,      # supervisory minimum used for breach flags (5%)
}


def project_capital(
    starting_position: pd.DataFrame,
    npl_paths: pd.DataFrame,
    assumptions: dict | None = None,
) -> pd.DataFrame:
    """
    starting_position: one row per bank with
        bank_id, npl_ratio, net_loans, total_assets, equity
    npl_paths: output of SatelliteNPLModel.project()
    Returns bank-quarter projection with losses, equity, ratio, breach flag.
    """
    a = {**DEFAULTS, **(assumptions or {})}
    rows = []

    for bank_id, path in npl_paths.groupby("bank_id"):
        start = starting_position.loc[
            starting_position["bank_id"] == bank_id
        ].iloc[0]
        equity = float(start["equity"])
        loans = float(start["net_loans"])
        assets = float(start["total_assets"])
        prev_npl = float(start["npl_ratio"])

        for _, q in path.sort_values("date").iterrows():
            d_npl = max(q["npl_ratio"] - prev_npl, 0.0) / 100.0
            credit_loss = d_npl * loans * a["lgd"]
            ppnr = a["ppnr_roa_q"] * assets
            net_income = ppnr - credit_loss
            equity += net_income
            ratio = equity / assets
            rows.append(
                {
                    "bank_id": bank_id,
                    "date": q["date"],
                    "npl_ratio": q["npl_ratio"],
                    "credit_loss": credit_loss,
                    "ppnr": ppnr,
                    "net_income": net_income,
                    "equity": equity,
                    "capital_ratio": ratio,
                    "breach": ratio < a["min_ratio"],
                }
            )
            prev_npl = q["npl_ratio"]

    return pd.DataFrame(rows)


def summarize(projection: pd.DataFrame, bank_names: dict | None = None) -> pd.DataFrame:
    """End-of-horizon summary per bank: trough ratio, total losses, breach."""
    out = (
        projection.groupby("bank_id")
        .agg(
            trough_ratio=("capital_ratio", "min"),
            end_ratio=("capital_ratio", "last"),
            total_credit_loss=("credit_loss", "sum"),
            breach=("breach", "any"),
        )
        .reset_index()
        .sort_values("trough_ratio")
    )
    if bank_names:
        out.insert(1, "bank", out["bank_id"].map(bank_names))
    return out
