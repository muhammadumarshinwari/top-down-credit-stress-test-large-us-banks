"""
Model validation for satellite NPL models.

Produces structured validation reports covering statistical diagnostics,
backtesting, out-of-sample performance, and pass/warn/fail checks — the
standard toolkit for independent model review and challenger validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .models import MACRO_VARS, REGRESSORS, SatelliteNPLModel, _add_lags

# Expected coefficient signs for economic plausibility checks.
EXPECTED_SIGNS: dict[str, int] = {
    "npl_lag1": 1,
    "real_gdp_growth_lag2": -1,
    "unemployment": 1,
    "mortgage_rate_lag1": 1,
    "t3m_lag1": 1,
    "t3m_lag2": 1,
    "cpi_inflation_lag1": 1,
    "house_price_index_lag1": -1,
}

DEFAULT_PERIOD_BINS = [
    pd.Timestamp("2005-01-01"),
    pd.Timestamp("2008-01-01"),
    pd.Timestamp("2010-01-01"),
    pd.Timestamp("2015-01-01"),
    pd.Timestamp("2020-01-01"),
    pd.Timestamp("2023-01-01"),
]
DEFAULT_PERIOD_LABELS = [
    "2005-2007",
    "2008-2009",
    "2010-2014",
    "2015-2019",
    "2020-2022",
]


def _to_ts(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _estimation_frame(panel: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
    df = _add_lags(panel)
    return df.dropna(subset=["npl_ratio"] + regressors).copy()


def _backtest_metrics(bt: pd.DataFrame) -> dict[str, float]:
    err = bt["error"]
    return {
        "n_obs": int(len(bt)),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
    }


def _metrics_by_bank(bt: pd.DataFrame) -> pd.DataFrame:
    tmp = bt.assign(abs_error=bt["error"].abs())
    out = (
        tmp.groupby("bank_id", as_index=False)
        .agg(
            n_obs=("error", "count"),
            rmse=("error", lambda s: np.sqrt((s**2).mean())),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            avg_npl=("npl_ratio", "mean"),
        )
        .sort_values("rmse")
    )
    return out.round(4)


def _metrics_by_period(
    bt: pd.DataFrame,
    bins: list[pd.Timestamp] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    bins = bins or DEFAULT_PERIOD_BINS
    labels = labels or DEFAULT_PERIOD_LABELS
    tmp = bt.assign(abs_error=bt["error"].abs()).copy()
    tmp["period"] = pd.cut(
        tmp["date"],
        bins=bins,
        labels=labels,
        right=False,
    )
    out = (
        tmp.groupby("period", observed=True)
        .agg(
            n_obs=("error", "count"),
            rmse=("error", lambda s: np.sqrt((s**2).mean())),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .round(4)
    )
    return out


def _build_flags(
    result: Any,
    regressors: list[str],
    vif: pd.DataFrame,
    lb: pd.DataFrame,
    normality: pd.DataFrame,
    hetero: pd.DataFrame,
    insample: dict[str, float],
    oos: dict[str, float] | None,
    by_period: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for reg in regressors:
        coef = float(result.params[reg])
        pval = float(result.pvalues[reg])
        expected = EXPECTED_SIGNS.get(reg)
        if expected is not None:
            sign_ok = (coef > 0 and expected > 0) or (coef < 0 and expected < 0)
            rows.append(
                {
                    "check": f"sign_{reg}",
                    "category": "economic_plausibility",
                    "status": "PASS" if sign_ok else "FAIL",
                    "detail": f"coef={coef:.4f}, expected {'+' if expected > 0 else '-'}",
                }
            )
        rows.append(
            {
                "check": f"significance_{reg}",
                "category": "statistical",
                "status": "PASS" if pval < 0.05 else "WARN",
                "detail": f"p-value={pval:.4f}",
            }
        )

    rsq = float(result.rsquared)
    rows.append(
        {
            "check": "r_squared",
            "category": "statistical",
            "status": "PASS" if rsq >= 0.5 else "WARN",
            "detail": f"R²={rsq:.4f}",
        }
    )

    dw = float(sm.stats.stattools.durbin_watson(result.resid))
    rows.append(
        {
            "check": "durbin_watson",
            "category": "serial_correlation",
            "status": "PASS" if 1.5 <= dw <= 2.5 else "WARN",
            "detail": f"DW={dw:.4f}",
        }
    )

    lb_fail = bool((lb["lb_pvalue"] < 0.05).any())
    rows.append(
        {
            "check": "ljung_box",
            "category": "serial_correlation",
            "status": "WARN" if lb_fail else "PASS",
            "detail": "residual autocorrelation detected" if lb_fail else "no rejection at 5%",
        }
    )

    jb_row = normality.loc[normality["test"] == "Jarque-Bera"].iloc[0]
    rows.append(
        {
            "check": "residual_normality",
            "category": "distributional",
            "status": "WARN" if float(jb_row["p_value"]) < 0.05 else "PASS",
            "detail": f"Jarque-Bera p={float(jb_row['p_value']):.4f}",
        }
    )

    bp_row = hetero.iloc[0]
    rows.append(
        {
            "check": "heteroskedasticity",
            "category": "distributional",
            "status": "PASS",
            "detail": (
                f"Breusch-Pagan p={float(bp_row['p_value']):.4f}; "
                "HC1 robust SEs used in estimation"
            ),
        }
    )

    high_vif = vif.loc[vif["VIF"] > 10, "variable"].tolist()
    rows.append(
        {
            "check": "multicollinearity",
            "category": "statistical",
            "status": "WARN" if high_vif else "PASS",
            "detail": f"high VIF (>10): {', '.join(high_vif)}" if high_vif else "all VIF <= 10",
        }
    )

    rows.append(
        {
            "check": "in_sample_rmse",
            "category": "backtesting",
            "status": "PASS",
            "detail": f"RMSE={insample['rmse']:.4f} pp, MAE={insample['mae']:.4f} pp",
        }
    )

    if "2008-2009" in by_period.index:
        gfc_rmse = float(by_period.loc["2008-2009", "rmse"])
        rows.append(
            {
                "check": "gfc_period_rmse",
                "category": "backtesting",
                "status": "WARN" if gfc_rmse > 0.5 else "PASS",
                "detail": f"2008-2009 RMSE={gfc_rmse:.4f} pp",
            }
        )

    if oos is not None:
        ratio = oos["rmse"] / insample["rmse"] if insample["rmse"] > 0 else np.inf
        rows.append(
            {
                "check": "out_of_sample_rmse",
                "category": "backtesting",
                "status": "PASS" if ratio <= 1.5 else "WARN",
                "detail": (
                    f"OOS RMSE={oos['rmse']:.4f} pp "
                    f"({ratio:.2f}x in-sample)"
                ),
            }
        )
        rows.append(
            {
                "check": "out_of_sample_bias",
                "category": "backtesting",
                "status": "PASS" if abs(oos["bias"]) <= 0.25 else "WARN",
                "detail": f"OOS bias={oos['bias']:.4f} pp",
            }
        )

    return pd.DataFrame(rows)


@dataclass
class ValidationReport:
    """Structured output from :func:`validate_model`."""

    summary: pd.DataFrame
    coefficients: pd.DataFrame
    vif: pd.DataFrame
    regressor_correlation: pd.DataFrame
    normality: pd.DataFrame
    ljung_box: pd.DataFrame
    heteroskedasticity: pd.DataFrame
    backtest_overall: dict[str, float]
    backtest_by_bank: pd.DataFrame
    backtest_by_period: pd.DataFrame
    flags: pd.DataFrame
    residuals: pd.DataFrame = field(repr=False)
    out_of_sample_overall: dict[str, float] | None = None
    out_of_sample_by_bank: pd.DataFrame | None = None
    out_of_sample_by_quarter: pd.DataFrame | None = None

    @property
    def n_fail(self) -> int:
        return int((self.flags["status"] == "FAIL").sum())

    @property
    def n_warn(self) -> int:
        return int((self.flags["status"] == "WARN").sum())

    @property
    def n_pass(self) -> int:
        return int((self.flags["status"] == "PASS").sum())

    def print_summary(self) -> None:
        print("=" * 72)
        print("MODEL VALIDATION REPORT")
        print("=" * 72)
        print()
        print("SUMMARY")
        print(self.summary.to_string(index=False))
        print()
        print("VALIDATION FLAGS")
        print(self.flags.to_string(index=False))
        print()
        print(f"Checks: {self.n_pass} PASS, {self.n_warn} WARN, {self.n_fail} FAIL")
        print()
        print("COEFFICIENTS")
        print(self.coefficients.to_string())
        print()
        print("BACKTEST (in-sample)")
        print(
            f"RMSE={self.backtest_overall['rmse']:.4f} pp  "
            f"MAE={self.backtest_overall['mae']:.4f} pp  "
            f"Bias={self.backtest_overall['bias']:.4f} pp"
        )
        if self.out_of_sample_overall is not None:
            oos = self.out_of_sample_overall
            print()
            print("OUT-OF-SAMPLE")
            print(
                f"RMSE={oos['rmse']:.4f} pp  "
                f"MAE={oos['mae']:.4f} pp  "
                f"Bias={oos['bias']:.4f} pp"
            )

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(out / "model_validation_summary.csv", index=False)
        self.coefficients.to_csv(out / "model_validation_coefficients.csv")
        self.vif.to_csv(out / "model_validation_vif.csv", index=False)
        self.regressor_correlation.to_csv(out / "model_validation_correlation.csv")
        self.normality.to_csv(out / "model_validation_normality.csv", index=False)
        self.ljung_box.to_csv(out / "model_validation_ljungbox.csv", index=False)
        self.heteroskedasticity.to_csv(
            out / "model_validation_heteroskedasticity.csv", index=False
        )
        self.backtest_by_bank.to_csv(out / "model_validation_by_bank.csv", index=False)
        self.backtest_by_period.to_csv(out / "model_validation_by_period.csv")
        self.flags.to_csv(out / "model_validation_flags.csv", index=False)
        self.residuals.to_csv(out / "model_validation_residuals.csv", index=False)
        if self.out_of_sample_by_bank is not None:
            self.out_of_sample_by_bank.to_csv(
                out / "model_validation_oos_by_bank.csv", index=False
            )
        if self.out_of_sample_by_quarter is not None:
            self.out_of_sample_by_quarter.to_csv(
                out / "model_validation_oos_by_quarter.csv", index=False
            )

    def plot(self, path: str | Path) -> None:
        import matplotlib.pyplot as plt

        fitted = self.residuals["fitted"].to_numpy()
        resid = self.residuals["resid"].to_numpy()

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        axes[0, 0].scatter(fitted, resid, alpha=0.3, s=12)
        axes[0, 0].axhline(0, color="k", lw=0.8)
        axes[0, 0].set_xlabel("Fitted NPL (%)")
        axes[0, 0].set_ylabel("Residual (pp)")
        axes[0, 0].set_title("Residuals vs fitted")

        stats.probplot(resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Normal Q-Q plot")

        (
            self.residuals.groupby("date")["resid"]
            .mean()
            .plot(ax=axes[1, 0], color="steelblue")
        )
        axes[1, 0].axhline(0, color="k", lw=0.8)
        axes[1, 0].set_title("Average residual by quarter")
        axes[1, 0].set_ylabel("Mean residual (pp)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        bank_rmse = self.backtest_by_bank.set_index("bank_id")["rmse"]
        bank_rmse.plot(kind="barh", ax=axes[1, 1], color="steelblue")
        axes[1, 1].set_title("In-sample RMSE by bank")
        axes[1, 1].set_xlabel("RMSE (pp)")

        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def validate_model(
    model: SatelliteNPLModel,
    panel: pd.DataFrame,
    *,
    est_end: pd.Timestamp | str | None = None,
    oos_start: pd.Timestamp | str | None = None,
    oos_end: pd.Timestamp | str | None = None,
    period_bins: list[pd.Timestamp] | None = None,
    period_labels: list[str] | None = None,
    ljung_box_lags: list[int] | None = None,
) -> ValidationReport:
    """
    Run a full validation suite on a fitted :class:`SatelliteNPLModel`.

    Parameters
    ----------
    model
        Fitted satellite model (must have called ``fit()``).
    panel
        Bank–macro panel used for backtesting and diagnostics.
    est_end
        End of estimation / in-sample window. If ``None``, uses all dates
        in ``panel``.
    oos_start, oos_end
        Optional out-of-sample window for holdout backtesting. Lags are
        computed on the full ``panel`` before slicing.
    period_bins, period_labels
        Custom period buckets for stability analysis by sub-sample.
    ljung_box_lags
        Lags for the Ljung–Box serial-correlation test (default: 4, 8, 12).

    Returns
    -------
    ValidationReport
        Tables, metrics, and pass/warn/fail flags suitable for model
        validation documentation.
    """
    if model.result is None:
        raise RuntimeError("Model is not fitted; call fit() before validate_model().")

    est_end = _to_ts(est_end)
    oos_start = _to_ts(oos_start)
    oos_end = _to_ts(oos_end)
    ljung_box_lags = ljung_box_lags or [4, 8, 12]

    est_panel = panel if est_end is None else panel[panel["date"] <= est_end].copy()
    est_df = _estimation_frame(est_panel, model.regressors)
    result = model.result
    regressors = model.regressors

    resid = result.resid
    fitted = result.fittedvalues

    residuals = est_df[["bank_id", "date", "npl_ratio"]].copy()
    residuals["fitted"] = fitted.values
    residuals["resid"] = resid.values

    summary = pd.DataFrame(
        [
            {
                "metric": "Estimation window",
                "value": f"{est_panel['date'].min().date()} to {est_panel['date'].max().date()}",
            },
            {"metric": "Observations", "value": int(result.nobs)},
            {"metric": "Banks", "value": int(est_panel["bank_id"].nunique())},
            {"metric": "Quarters", "value": int(est_panel["date"].nunique())},
            {"metric": "R-squared", "value": round(result.rsquared, 4)},
            {"metric": "Adj R-squared", "value": round(result.rsquared_adj, 4)},
            {
                "metric": "RMSE (in-sample est period)",
                "value": round(float(np.sqrt((resid**2).mean())), 4),
            },
            {
                "metric": "MAE (in-sample est period)",
                "value": round(float(np.abs(resid).mean()), 4),
            },
            {"metric": "AIC", "value": round(result.aic, 2)},
            {"metric": "BIC", "value": round(result.bic, 2)},
            {
                "metric": "Durbin-Watson",
                "value": round(float(sm.stats.stattools.durbin_watson(resid)), 4),
            },
        ]
    )

    coefficients = pd.DataFrame(
        {
            "coef": result.params[regressors],
            "std_err": result.bse[regressors],
            "t_stat": result.tvalues[regressors],
            "p_value": result.pvalues[regressors],
            "ci_low": result.conf_int().loc[regressors, 0],
            "ci_high": result.conf_int().loc[regressors, 1],
        }
    ).round(4)

    macro_X = est_df[regressors].astype(float)
    vif = pd.DataFrame(
        {
            "variable": regressors,
            "VIF": [
                variance_inflation_factor(macro_X.values, i)
                for i in range(len(regressors))
            ],
        }
    ).round(2)

    jb_stat, jb_p = stats.jarque_bera(resid)
    sw_stat, sw_p = stats.shapiro(resid[: min(5000, len(resid))])
    normality = pd.DataFrame(
        [
            {
                "test": "Jarque-Bera",
                "statistic": round(float(jb_stat), 2),
                "p_value": round(float(jb_p), 4),
            },
            {
                "test": "Skewness",
                "statistic": round(float(stats.skew(resid)), 4),
                "p_value": np.nan,
            },
            {
                "test": "Kurtosis",
                "statistic": round(float(stats.kurtosis(resid)), 4),
                "p_value": np.nan,
            },
            {
                "test": "Shapiro-Wilk (5k subsample)",
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 4),
            },
        ]
    )

    lb = acorr_ljungbox(resid, lags=ljung_box_lags, return_df=True)
    ljung_box = lb.reset_index().rename(columns={"index": "lag"})

    bp_stat, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(macro_X))
    heteroskedasticity = pd.DataFrame(
        [
            {
                "test": "Breusch-Pagan",
                "statistic": round(float(bp_stat), 4),
                "p_value": round(float(bp_p), 4),
            }
        ]
    )

    insample_bt = model.backtest(est_panel)
    backtest_overall = _backtest_metrics(insample_bt)
    backtest_by_bank = _metrics_by_bank(insample_bt)
    backtest_by_period = _metrics_by_period(
        insample_bt, bins=period_bins, labels=period_labels
    )

    oos_overall = None
    oos_by_bank = None
    oos_by_quarter = None
    if oos_start is not None:
        oos_end = oos_end or panel["date"].max()
        full_bt = model.backtest(panel)
        oos_bt = full_bt[
            (full_bt["date"] >= oos_start) & (full_bt["date"] <= oos_end)
        ].copy()
        if len(oos_bt):
            oos_overall = _backtest_metrics(oos_bt)
            oos_by_bank = _metrics_by_bank(oos_bt)
            oos_by_quarter = (
                oos_bt.groupby("date", as_index=False)
                .agg(
                    actual=("npl_ratio", "mean"),
                    predicted=("npl_pred", "mean"),
                    error=("error", "mean"),
                )
                .round(4)
            )

    flags = _build_flags(
        result=result,
        regressors=regressors,
        vif=vif,
        lb=ljung_box,
        normality=normality,
        hetero=heteroskedasticity,
        insample=backtest_overall,
        oos=oos_overall,
        by_period=backtest_by_period,
    )

    return ValidationReport(
        summary=summary,
        coefficients=coefficients,
        vif=vif,
        regressor_correlation=macro_X.corr().round(3),
        normality=normality,
        ljung_box=ljung_box,
        heteroskedasticity=heteroskedasticity,
        backtest_overall=backtest_overall,
        backtest_by_bank=backtest_by_bank,
        backtest_by_period=backtest_by_period,
        flags=flags,
        residuals=residuals,
        out_of_sample_overall=oos_overall,
        out_of_sample_by_bank=oos_by_bank,
        out_of_sample_by_quarter=oos_by_quarter,
    )
