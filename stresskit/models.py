"""
Satellite credit risk model.

Links bank-level credit quality (NPL ratio) to macro conditions, so that a
macro scenario can be translated into scenario-conditional NPL paths —
the same satellite-model logic used in central bank stress testing.

Specification (selected lags + contemporaneous unemployment):

    npl_ratio[i,t] = a_i
                     + rho  * npl_ratio[i,t-1]
                     + b1   * real_gdp_growth[t-2]
                     + b2   * unemployment[t]
                     + b3   * mortgage_rate[t-1] + e[i,t]

with bank fixed effects a_i. Estimated by OLS with bank dummies.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

MACRO_VARS = [
    "real_gdp_growth",
    "unemployment",
    "t3m",
    "cpi_inflation",
    "mortgage_rate",
    "house_price_index",
]

REGRESSORS = [
    "npl_lag1",
    "real_gdp_growth_lag2",
    "unemployment",
    "mortgage_rate_lag1",
]

SCENARIO_MACRO_VARS = MACRO_VARS


def _add_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach two lags of NPL (by bank) and macro variables (by date)."""
    df = panel.sort_values(["bank_id", "date"]).copy()
    df["npl_lag1"] = df.groupby("bank_id")["npl_ratio"].shift(1)
    df["npl_lag2"] = df.groupby("bank_id")["npl_ratio"].shift(2)

    macro = (
        df[["date"] + MACRO_VARS]
        .drop_duplicates("date")
        .sort_values("date")
    )
    for var in MACRO_VARS:
        macro[f"{var}_lag1"] = macro[var].shift(1)
        macro[f"{var}_lag2"] = macro[var].shift(2)

    lag_cols = ["date"] + [
        f"{var}_lag{lag}" for var in MACRO_VARS for lag in (1, 2)
    ]
    return df.merge(macro[lag_cols], on="date", how="left")


_add_gdp_lags = _add_lags


def _regressor_value(
    name: str,
    npl_hist: list[float],
    macro_hist: dict[str, list[float]],
    current: pd.Series | dict,
) -> float:
    if name in MACRO_VARS:
        return float(current[name])
    if name == "npl_lag1":
        return npl_hist[-1]
    if name == "npl_lag2":
        return npl_hist[-2]
    for var in MACRO_VARS:
        if name == f"{var}_lag1":
            return macro_hist[var][-1]
        if name == f"{var}_lag2":
            return macro_hist[var][-2]
    raise KeyError(f"Unknown regressor: {name}")


def _predict(
    p: pd.Series,
    fe: float,
    npl_hist: list[float],
    macro_hist: dict[str, list[float]],
    current: pd.Series | dict,
    regressors: list[str],
) -> float:
    pred = fe + sum(
        p[col] * _regressor_value(col, npl_hist, macro_hist, current)
        for col in regressors
    )
    return max(pred, 0.0)


class SatelliteNPLModel:
    def __init__(self, regressors: list[str] | None = None) -> None:
        self.regressors = regressors or REGRESSORS
        self.result = None
        self.fixed_effects: dict[int, float] = {}
        self._macro_history: pd.DataFrame | None = None
        self._npl_tail: dict[int, list[float]] = {}

    def fit(self, panel: pd.DataFrame) -> "SatelliteNPLModel":
        """panel: bank_id, date, npl_ratio, and all MACRO_VARS."""
        df = _add_lags(panel)
        df = df.dropna(subset=["npl_ratio"] + self.regressors)

        X = pd.get_dummies(df["bank_id"], prefix="fe", dtype=float)
        X = pd.concat(
            [df[self.regressors].reset_index(drop=True), X.reset_index(drop=True)],
            axis=1,
        )
        y = df["npl_ratio"].reset_index(drop=True)

        self.result = sm.OLS(y, X).fit(cov_type="HC1")
        self.fixed_effects = {
            int(c.split("_")[1]): self.result.params[c]
            for c in X.columns if c.startswith("fe_")
        }
        self._macro_history = (
            panel[["date"] + MACRO_VARS]
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
        self._npl_tail = (
            panel.sort_values(["bank_id", "date"])
            .groupby("bank_id")["npl_ratio"]
            .apply(lambda s: [float(x) for x in s.tail(2).tolist()])
            .to_dict()
        )
        return self

    def project(
        self, last_obs: pd.DataFrame, scenario: pd.DataFrame
    ) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call fit() before project().")
        if self._macro_history is None:
            raise RuntimeError("Macro history missing; call fit() before project().")

        p = self.result.params
        scenario = scenario.sort_values("date")
        macro_hist = {
            var: self._macro_history[var].astype(float).tolist()
            for var in MACRO_VARS
        }
        out = []

        for _, bank in last_obs.iterrows():
            bank_id = int(bank["bank_id"])
            fe = self.fixed_effects.get(bank_id, 0.0)
            npl_hist = list(self._npl_tail.get(bank_id, [float(bank["npl_ratio"])] * 2))
            if len(npl_hist) == 1:
                npl_hist = [npl_hist[0], npl_hist[0]]

            for _, q in scenario.iterrows():
                npl = _predict(p, fe, npl_hist, macro_hist, q, self.regressors)
                out.append(
                    {"bank_id": bank_id, "date": q["date"], "npl_ratio": npl}
                )
                npl_hist.append(npl)
                for var in MACRO_VARS:
                    macro_hist[var].append(float(q[var]))

        return pd.DataFrame(out)

    def backtest(self, panel: pd.DataFrame) -> pd.DataFrame:
        """One-step-ahead in-sample fit: predicted vs actual NPL, with errors."""
        df = _add_lags(panel)
        df = df.dropna(subset=["npl_ratio"] + self.regressors)
        p = self.result.params
        rows = []
        for _, row in df.iterrows():
            npl_hist = [row["npl_lag2"], row["npl_lag1"]]
            macro_hist = {
                var: [row[f"{var}_lag2"], row[f"{var}_lag1"]]
                for var in MACRO_VARS
            }
            fe = self.fixed_effects.get(int(row["bank_id"]), 0.0)
            rows.append(_predict(p, fe, npl_hist, macro_hist, row, self.regressors))
        df["npl_pred"] = rows
        df["error"] = df["npl_pred"] - df["npl_ratio"]
        return df[["bank_id", "date", "npl_ratio", "npl_pred", "error"]]

    def validate(
        self,
        panel: pd.DataFrame,
        **kwargs,
    ):
        """Run the model validation suite; see :func:`validation.validate_model`."""
        from .validation import validate_model

        return validate_model(self, panel, **kwargs)
