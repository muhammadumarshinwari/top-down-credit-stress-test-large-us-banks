# Top-Down Credit Stress Test for Large U.S. Banks

> A reduced-form macro stress test of credit risk across thirteen large US banks. Built as a portfolio project to demonstrate satellite modeling, validation, and capital projection methodology used in supervisory stress testing.

**Methods:** panel fixed-effects regression 
**Data:** FDIC BankFind API · FRED · Fed DFAST scenario files
**Stack:** Python · statsmodels · pandas · matplotlib

---

## Headline result

The 2025 Fed severely adverse scenario applied to thirteen large US banks over a nine-quarter horizon:

<table>
<tr><th align="left">Stress test outcome</th><th></th><th align="left">Model fit</th><th></th></tr>
<tr><td>Total system credit loss</td><td align="right"><b>$38.7B</b></td><td>In-sample RMSE</td><td align="right"><b>0.31 pp</b></td></tr>
<tr><td>Median trough Tier 1 ratio</td><td align="right"><b>13.0%</b></td><td>Out-of-sample RMSE</td><td align="right"><b>0.16 pp</b></td></tr>
<tr><td>Most stressed bank (trough)</td><td align="right"><b>Regions, 11.4%</b></td><td>R-squared</td><td align="right"><b>0.968</b></td></tr>
<tr><td>Least stressed bank (trough)</td><td align="right"><b>JPMorgan, 16.2%</b></td><td>Validation flags</td><td align="right"><b>13 PASS / 5 WARN</b></td></tr>
<tr><td>Banks breaching 8.5% Tier 1 floor</td><td align="right"><b>0 of 13</b></td><td>Observations</td><td align="right"><b>1,014</b></td></tr>
</table>

Capital metric is Tier 1 / RWA (a CET1 proxy — see [Stress test results](#stress-test-results)). The 8.5% threshold is the 6% Tier 1 minimum plus a 2.5% conservation buffer. All numbers reproduce from `python examples/run_all.py`.

---

## Pipeline

```
FDIC call reports      FRED macro history       2025 Fed scenario
       │                       │                        │
       └───────────┬───────────┘                        │
                   ▼                                    │
        Estimation panel (1,014 obs)                    │
                   │                                    │
                   ▼                                    │
        Satellite NPL model                             │
        (panel FE + AR(1) + macro)                      │
                   │                                    │
                   ▼                                    │
        Validation suite (13 PASS, 5 WARN)              │
                   │                                    │
                   └───────────┬────────────────────────┘
                               ▼
                  Nine-quarter NPL projection
                               │
                               ▼
                  Capital roll-forward
                  (NPL → losses → equity → ratio)
```

---

## How banks actually stress credit (and what this project does instead)

Walk into a large bank's risk team and ask how they stress credit. You will hear about **probability of default (PD)**, **loss given default (LGD)**, and **exposure at default (EAD)**, often modeled at the loan or facility level with rating migrations and IFRS 9 stage allocation driving provisions. CCAR and DFAST go further: segment-level models for CRE, cards, mortgages, and C&I, with collateral and prepayment dynamics, all shocked under a supervisory macro scenario. That bottom-up stack is the production system.

This project is **not** that. It is a **top-down satellite model** of the kind supervisors and central banks use when they need to link a macro scenario to bank-level credit outcomes without full loan-level data. The NPL ratio enters directly as a function of lagged NPL and a small set of macro variables. Losses are backed out with a fixed LGD. The capital module rolls equity forward from pre-provision income minus credit losses. All inputs are public.

---

## The workflow

1. **Build the panel.** Pull quarterly financials for thirteen US banks from the FDIC and merge with FRED macro history.
2. **Estimate the satellite.** Regress each bank's NPL ratio on its own lag, GDP growth, unemployment, and mortgage rates, with bank fixed effects.
3. **Validate.** Backtest in sample, forecast out of sample on 2023-2024, run standard regression diagnostics, and check unit roots.
4. **Project.** Feed the 2025 Fed severely adverse scenario through the model, convert NPL paths into losses, and roll capital forward.

---

## Data

### Banks and the credit metric

Thirteen large US banks, all of which are subsidiaries of bank holding companies covered by the Fed's 2025 DFAST exercise. Quarterly data from the [FDIC BankFind API](https://banks.data.fdic.gov/docs/) (free, no key). After the first download, results are cached locally.

The dependent variable is the **NPL ratio** from call report fields:

$$\text{NPL ratio (\\%)} = \frac{\text{NCLNLS}}{\text{LNLSNET}} \times 100$$

`NCLNLS` is noncurrent loans and leases. `LNLSNET` is net loans and leases. A coarse but standard portfolio-level credit quality measure.

| CERT | Bank | Parent BHC (DFAST entity) |
|---:|---|---|
| 628 | JPMorgan Chase Bank NA | JPMorgan Chase & Co. |
| 3510 | Bank of America NA | Bank of America Corporation |
| 3511 | Wells Fargo Bank NA | Wells Fargo & Company |
| 7213 | Citibank NA | Citigroup Inc. |
| 6548 | U.S. Bank NA | U.S. Bancorp |
| 6384 | PNC Bank NA | PNC Financial Services Group |
| 9846 | Truist Bank | Truist Financial Corporation |
| 6672 | Fifth Third Bank NA | Fifth Third Bancorp |
| 17534 | KeyBank NA | KeyCorp |
| 12368 | Regions Bank | Regions Financial Corporation |
| 588 | M&T Bank | M&T Bank Corporation |
| 57957 | Citizens Bank NA | Citizens Financial Group |
| 6560 | The Huntington National Bank | Huntington Bancshares |

### Why these thirteen and not all twenty-two DFAST banks?

The Fed's 2025 DFAST covers twenty-two bank holding companies. The remaining nine are excluded from this panel because the satellite is a **credit risk model**, and their business models are not credit-driven in the way this specification assumes:

| Excluded BHC | Reason |
|---|---|
| Goldman Sachs Group | Broker-dealer dominant; bank subsidiary holds mostly cash and treasuries, not loans |
| Morgan Stanley | Same as Goldman — bank subsidiary is small relative to securities business |
| The Bank of New York Mellon | Custody bank; minimal loan book, NPL ratio is essentially zero |
| State Street Corporation | Custody bank — same profile as BNY Mellon |
| Northern Trust Corporation | Custody and wealth management; loan book is collateralized securities lending |
| The Charles Schwab Corporation | Wealth management and brokerage; bank arm is mainly margin and securities-backed lending |
| Capital One Financial | Credit card heavy; NPL dynamics and macro sensitivities are very different from commercial loans |
| American Express Company | Pure card portfolio; baseline NPL runs much higher and reacts differently |
| Ally Financial | Auto-loan dominant; different LGD assumption and macro betas |

The right way to bring these into the framework would be to estimate **separate satellites by portfolio type** — a card model, an auto model, a custody model — which is exactly how production CCAR / DFAST stacks are organized. That is the natural extension; see [Extension roadmap](#extension-roadmap).

### Macro variables

Bundled in `data/fred_macro_history.csv` so the project runs without a FRED API key. The final model uses four regressors; the panel includes a wider set for specification testing.

| Variable | FRED series | In final model | Lag |
|---|---|:---:|:---:|
| Real GDP growth | `A191RL1Q225SBEA` | Yes | t&minus;2 |
| Unemployment | `UNRATE` | Yes | t |
| 30-year mortgage rate | `MORTGAGE30US` | Yes | t&minus;1 |
| 3-month T-bill | `TB3MS` | No | — |
| CPI inflation | `CPIAUCSL` | No | — |
| House price index | `CSUSHPINSA` | No | — |

Panel: **2005 Q1 to 2024 Q4** (80 quarters per bank). After attaching two lags, **1,014 bank-quarters** remain for estimation. The validation window restricts to estimation through 2022 Q4 (910 obs) and reserves 2023-2024 for out-of-sample testing.

### Stress scenario

The **2025 Fed supervisory severely adverse domestic scenario** (`data/2025-Table_3A_Supervisory_Severely_Adverse_Domestic.csv`). Bank history ends 2024 Q4. Scenario path starts 2025 Q1 and runs nine quarters, matching the DFAST planning horizon.

---

## The satellite model

### Specification

$$
\text{NPL}_{i,t} = \alpha_i + \rho\,\text{NPL}_{i,t-1} + \beta_1\,\text{GDP}_{t-2} + \beta_2\,\text{UR}_{t} + \beta_3\,\text{Mtg}_{t-1} + \varepsilon_{i,t}
$$

The lagged NPL term captures persistence — credit quality does not reset every quarter. GDP enters at lag 2 (weaker growth shows up in delinquencies with a delay). Unemployment is contemporaneous. The mortgage rate enters with one lag, proxying for household debt-service pressure.

Estimation is pooled OLS with bank dummies and **HC1 robust standard errors**. Bank fixed effects only — no time fixed effects, so common macro shocks enter through the regressors rather than through time dummies.

### How I got to this specification

I did not start with four variables. The first pass included two lags of GDP, unemployment, and short rates on top of lagged NPL. That fit well in sample but was too rich for ten banks. I dropped variables and lags step by step, checking forecast stability and coefficient signs at each stage. GDP worked better at lag 2 than lag 1. Contemporaneous unemployment beat a lagged term. The mortgage rate added more than the T-bill alone. HPI, CPI, and the T-bill were dropped from the final spec.

That iterative trimming is normal in satellite model development. The goal is a stable, interpretable mapping from scenario to NPL that a validator can challenge — not the highest in-sample R-squared.

### Estimated coefficients (full panel, 1,014 obs.)

![Estimated coefficients with 95% confidence intervals](docs/coefficient_estimates.png)

**How to read.** Each bar is a regression coefficient. The horizontal whisker is its 95% confidence interval. Navy bars are positive (variables that push NPL up), burgundy bars are negative (variables that push NPL down). No interval crosses zero, meaning every coefficient is statistically significant.

| Variable | Coefficient | Std. err. | t-stat | p-value | Reading |
|---|--:|--:|--:|:---:|---|
| `npl_lag1` | **+0.938** | 0.010 | 92.0 | \*\*\* | NPL is highly persistent |
| `real_gdp_growth_lag2` | **&minus;0.008** | 0.002 | &minus;4.1 | \*\*\* | Weaker growth raises NPL with delay |
| `unemployment` | **+0.050** | 0.008 | 6.5 | \*\*\* | Higher unemployment raises NPL |
| `mortgage_rate_lag1` | **+0.069** | 0.008 | 9.1 | \*\*\* | Higher rates raise NPL with one-quarter lag |

<sub>\*\*\* p &lt; 0.01 &nbsp;·&nbsp; \*\* p &lt; 0.05 &nbsp;·&nbsp; \* p &lt; 0.10 &nbsp;·&nbsp; HC1 robust SEs &nbsp;·&nbsp; R² = 0.968 &nbsp;·&nbsp; N = 1,014</sub>

---

## Model validation

Validation matters as much as estimation. The project includes a `validate_model()` function that produces a structured report with automated PASS, WARN, and FAIL flags. I treat WARN seriously — a flag does not mean the model is broken, but I would discuss it in a validation memo.

### What the validator checks

Economic plausibility (do coefficient signs make sense?), statistical fit, multicollinearity, residual diagnostics, in-sample backtest error, out-of-sample forecast error, and a separate flag for GFC-period fit. Code in `stresskit/validation.py`.

```python
from stresskit import SatelliteNPLModel, validate_model

model = SatelliteNPLModel().fit(estimation_panel)
report = validate_model(
    model,
    full_panel,
    est_end="2022-12-31",
    oos_start="2023-03-31",
    oos_end="2024-12-31",
)
report.print_summary()
report.save("data/")
```

**Validation summary (estimation through 2022 Q4):** 13 PASS · 5 WARN · 0 FAIL

| Check | Status | Detail |
|---|:---:|---|
| Coefficient signs | PASS | All four regressors signed as theory suggests |
| Statistical significance | PASS | All p-values below 1% |
| R-squared | PASS | 0.968 |
| In-sample RMSE | PASS | 0.32 pp |
| Out-of-sample RMSE | PASS | 0.16 pp |
| Out-of-sample bias | PASS | +0.14 pp, model slightly under-predicts |
| Heteroskedasticity | PASS | Breusch-Pagan rejects, HC1 SEs used |
| Durbin-Watson | WARN | 1.14, some positive autocorrelation |
| Ljung-Box | WARN | Serial correlation in residuals |
| Residual normality | WARN | Fat tails, driven by the GFC |
| Multicollinearity | WARN | Unemployment VIF > 10 |
| GFC period RMSE | WARN | 0.73 pp in 2008-2009 vs &lt; 0.20 pp in calm periods |

The warnings are honest. A linear AR model with ten banks is not going to fit the GFC well, and unemployment correlates with lagged NPL by construction. In a bank validation I would document these points and test challenger specifications (crisis dummy, shorter estimation window, alternative unemployment transform). I would not hide them.

---

## Does the model track realized NPL?

### In-sample backtest

At each quarter, can the model predict that quarter's NPL using only information available up to the prior quarter? Overall error is **0.31 pp RMSE** and **0.18 pp MAE**.

| Bank | RMSE (pp) | MAE (pp) |
|---|---:|---:|
| Citizens | 0.17 | 0.12 |
| U.S. Bank | 0.19 | 0.13 |
| Regions | 0.21 | 0.16 |
| M&T Bank | 0.22 | 0.15 |
| Truist | 0.23 | 0.16 |
| KeyBank | 0.27 | 0.18 |
| Wells Fargo | 0.33 | 0.19 |
| JPMorgan Chase | 0.34 | 0.20 |
| Citibank | 0.34 | 0.21 |
| Fifth Third | 0.35 | 0.18 |
| Huntington | 0.39 | 0.21 |
| PNC | 0.40 | 0.17 |
| Bank of America | 0.40 | 0.26 |

![Backtest results](docs/backtest_results.png)

**How to read.** Top panel: each colored line is one bank, solid = actual NPL, dashed = one-step prediction. Where they stay close, the model is tracking realized credit quality. Bottom panel: average prediction error by quarter. Errors cluster around the Global Financial Crisis, when a linear model without a regime switch cannot capture the speed of deterioration.

**RMSE by sub-period:**

| Period | RMSE (pp) | Regime |
|---|---:|---|
| 2005-2007 | 0.21 | Pre-crisis |
| **2008-2009** | **0.73** | **Global Financial Crisis** |
| 2010-2014 | 0.29 | Post-crisis recovery |
| 2015-2019 | 0.12 | Calm expansion |
| 2020-2022 | 0.19 | COVID + reopening |

The GFC row is the one I would spend the most time on in a model review.

### Out-of-sample forecast (2023 Q1 to 2024 Q4)

In-sample fit can always be engineered. A more informative test: estimate through **2022 Q4**, forecast **2023 Q1 to 2024 Q4** (104 bank-quarters across 13 banks).

<table>
<tr><th>Metric</th><th align="right">Value</th><th>Read</th></tr>
<tr><td>RMSE</td><td align="right"><b>0.16 pp</b></td><td>Typical absolute miss is under 0.2 pp</td></tr>
<tr><td>MAE</td><td align="right"><b>0.14 pp</b></td><td>Median error in line with RMSE — few large outliers</td></tr>
<tr><td>Bias</td><td align="right"><b>+0.14 pp</b></td><td>Model systematically under-predicts realized NPL</td></tr>
</table>

NPL ticked up modestly in 2024 and the satellite, anchored on high persistence, lagged that turn slightly.

| Quarter | Actual NPL (%) | Predicted (%) | Error (pp) |
|---|---:|---:|---:|
| 2023 Q1 | 0.91 | 1.04 | +0.13 |
| 2023 Q2 | 0.88 | 1.02 | +0.14 |
| 2023 Q3 | 0.93 | 1.01 | +0.08 |
| 2023 Q4 | 0.99 | 1.12 | +0.13 |
| 2024 Q1 | 1.04 | 1.18 | +0.14 |
| 2024 Q2 | 1.01 | 1.19 | +0.18 |
| 2024 Q3 | 1.05 | 1.22 | +0.17 |
| 2024 Q4 | 1.07 | 1.18 | +0.12 |

![Out-of-sample forecast](docs/forecast_sample_results.png)

**How to read.** Top: actual (solid) vs predicted (dashed) NPL in the holdout window. The lines track each other more closely than during the GFC, which is why OOS RMSE is lower than full-sample in-sample RMSE. Bottom: quarter-by-quarter bias. The flip from positive to negative bias in 2024 is worth flagging in a validation report even though overall RMSE looks good.

### Regression diagnostics

Standard econometric health check on residuals from the 2022 Q4 estimation window.

![Model diagnostics](docs/model_diagnostics.png)

- **Residuals vs fitted:** Should scatter randomly around zero. Breusch-Pagan rejects homoskedasticity, which is why HC1 robust standard errors are used.
- **Q-Q plot:** Upper tail deviates from the diagonal — the GFC fat tail. Jarque-Bera rejects normality.
- **Average residual by quarter:** Systematic errors in particular periods. The GFC stands out again.
- **RMSE by bank:** Bank of America and PNC have the largest in-sample errors, consistent with the backtest table.

| Variable | VIF | Status |
|---|---:|:---|
| `npl_lag1` | 3.9 | Low |
| `real_gdp_growth_lag2` | 1.1 | Low |
| `unemployment` | 10.7 | Elevated multicollinearity (above 10) |
| `mortgage_rate_lag1` | 6.4 | Monitor (in 5–10 range) |

<sub>Conventional thresholds: VIF &lt; 5 low concern · 5–10 monitor · &gt; 10 elevated multicollinearity.</sub>

Unemployment VIF above 10 reflects correlation with lagged NPL. I would monitor coefficient stability across rolling windows before relying on that point estimate in a live stress test.

### Unit root tests

Before relying on NPL in levels I checked stationarity with Phillips-Perron and Zivot-Andrews tests (`arch` package). Bank NPL ratios do not reject a unit root in levels for most institutions. GDP growth is stationary. Unemployment is borderline.

This supports including the AR term. First-differencing NPL would remove the level information stress testers care about. Supervisors want the projected **level** of NPL under an adverse path, not just the change.

---

## Stress test results

I feed the **2025 Fed severely adverse scenario** through the satellite starting from each bank's 2024 Q4 position. The satellite produces a nine-quarter NPL path. The capital module converts NPL increases into losses and rolls **Tier 1 regulatory capital** forward against **risk-weighted assets**.

```
new NPLs (dollars)   = max(change in NPL ratio, 0) x net loans
credit loss          = new NPLs x LGD (45%)
net income           = PPNR - credit loss
tier 1 capital       = tier 1 capital + net income
capital ratio        = tier 1 capital / RWA
```

PPNR is a flat quarterly ROA (0.10% under stress vs 0.30% baseline). RWA is held flat through the horizon. No dividends, no capital raises.

### Why Tier 1 / RWA and not CET1 directly

CET1 / RWA is the headline ratio in CCAR / DFAST scoring. The FDIC BankFind API silently drops the CET1 fields (a known quirk — same issue we hit with the NPL ratio field), so I use **Tier 1 / RWA** as a CET1 proxy. For the ten large US banks in this panel, CET1 and Tier 1 differ by less than 0.5 percentage points across the entire post-Basel-III history, because they hold negligible non-CET1 Tier 1 instruments (e.g., AT1). The methodology, the projection logic, and the policy interpretation are identical.

The **8.5% threshold** is the regulatory 6% Tier 1 minimum plus the 2.5% capital conservation buffer. Below that, banks face restrictions on dividends and discretionary bonuses.

| Bank | Starting T1 (Q4 2024) | Trough T1 | End T1 | Total credit loss | Breach 8.5%? |
|---|---:|---:|---:|---:|:---:|
| Regions | 11.32% | 11.45% | 12.53% | $0.5B | No |
| PNC | 11.85% | 11.98% | 13.11% | $2.0B | No |
| M&T Bank | 12.32% | 12.39% | 13.56% | $0.8B | No |
| Citizens | 12.27% | 12.41% | 13.63% | $0.6B | No |
| Huntington | 12.39% | 12.53% | 13.75% | $0.7B | No |
| Truist | 12.61% | 12.74% | 13.85% | $1.8B | No |
| Fifth Third | 12.86% | 12.99% | 14.15% | $0.6B | No |
| KeyBank | 12.94% | 13.07% | 14.32% | $0.5B | No |
| Wells Fargo | 13.08% | 13.24% | 14.49% | $6.5B | No |
| Bank of America | 13.46% | 13.64% | 15.19% | $8.7B | No |
| U.S. Bank | 13.60% | 13.75% | 15.08% | $2.1B | No |
| Citibank | 14.03% | 14.18% | 15.66% | $3.9B | No |
| JPMorgan Chase | 16.04% | 16.24% | 18.08% | $10.0B | No |

<sub>Starting Tier 1 ratios are from FDIC call reports as of 2024 Q4 (pre-stress).</sub>

### Why these numbers do not match the Fed's published DFAST starting ratios

The Fed's [June 2025 DFAST results](https://www.federalreserve.gov/publications/2025-june-dodd-frank-act-stress-test-results.htm) report a starting capital ratio for each bank, and those numbers differ slightly from the starting Tier 1 ratios in the table above. Three reasons:

1. **Reporting entity.** FDIC call reports are filed at the **insured bank subsidiary** level (e.g., JPMorgan Chase Bank NA, CERT 628). The Fed publishes ratios for the **bank holding company** (JPMorgan Chase & Co.), which consolidates the bank plus broker-dealer, credit card, asset management, and insurance subsidiaries. BHC capital data comes from the FR Y-9C filing, not the call report. The two entities have different assets, RWA, and ratios.
2. **CET1 vs Tier 1.** The Fed's headline ratio is CET1 / RWA. The FDIC API silently drops the CET1 field, so this project uses **Tier 1 / RWA** as a CET1 proxy. CET1 is always less than or equal to Tier 1; the gap is additional Tier 1 instruments (preferred stock, AT1). For these banks the gap is typically 0.5 - 1.5 pp, so the Tier 1 ratio reported here runs slightly above the published CET1 ratio.
3. **Standardized vs Advanced approaches.** Large US banks compute ratios under both frameworks and the binding ratio is the lower of the two. The FDIC field used here is the standardized approach. The Fed publishes whichever approach is binding for each bank, which is sometimes the advanced internal-ratings-based number.

The project deliberately uses bank-subsidiary data because the FDIC BankFind API is fully public and reproducible without registration. Replicating DFAST inputs exactly would require BHC-level FR Y-9C data, which is a natural extension.

![Stress test results](docs/stress_results.png)

**How to read.** Left: projected NPL ratios. NPL roughly doubles over the horizon for most banks — the satellite's response to the macro path (rising unemployment, falling GDP, higher rates). Right: Tier 1 capital ratio. The red dashed line is the 8.5% effective floor (6% minimum + 2.5% conservation buffer); the gray dotted line is the bare 6% Tier 1 minimum. Capital ratios drift up because PPNR exceeds losses under these assumptions — a known simplification, not a finding about real bank resilience. A full DFAST-style projection would include RWA growth, balance sheet rebalancing, and often declining PPNR under stress.

In a real bank this stage would sit on top of hundreds of PD/LGD models by segment, with provisions flowing through the ECL framework and capital measured against full CET1, Tier 1 leverage, and SLR ratios. Here the point is to show that the satellite output connects to a regulatory capital narrative, not to replicate CCAR.

---

## Extension roadmap

If I were taking this from a portfolio project to a production-style exercise, these would be my priorities:

1. **Replace the NPL satellite with a PD-based segment structure** (CRE, cards, C&I, residential) if loan-level or segment data were available.
2. **Project RWA endogenously** under stress (right now RWA is held flat). Risk weights drift up as credit deteriorates.
3. **Model PPNR properly** with an NII satellite using repricing gaps and a deposit beta, instead of a flat ROA.
4. **Add a crisis regime** (dummy or threshold) so the GFC is not treated like a normal draw.
5. **Expand the bank sample** or stratify by asset size and business model.

Every simplification above is visible in the code as a named parameter. That is intentional. Stress testing is as much about assumptions as it is about econometrics.

---

## Running the project

Requires Python 3.10+ and the packages in `requirements.txt`. The first run downloads FDIC data over the internet and caches it locally.

```bash
git clone https://github.com/muhammadumarshinwari/top-down-credit-stress-test-large-us-banks.git
cd top-down-credit-stress-test-large-us-banks
pip install -r requirements.txt
python examples/run_all.py
```

That single command runs the stress test, out-of-sample forecast, validation suite, unit root tests, and PDF report. Each stage can also be run separately from the `examples/` folder. Key outputs land in `data/` (CSVs) and `docs/` (charts and `US_Bank_Credit_Stress_Test.pdf`).

---

*Educational project built entirely on public data. Not supervisory software; outputs are not assessments of any real bank's credit quality, provisions, or capital adequacy.*
