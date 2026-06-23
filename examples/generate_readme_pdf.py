"""
Generate the project PDF report (narrative + charts + tables).

Run:  python examples/generate_readme_pdf.py
Output: docs/US_Bank_Credit_Stress_Test.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"
OUT = DOCS / "US_Bank_Credit_Stress_Test.pdf"

NAVY = (20, 60, 120)
STEEL = (70, 130, 180)
HEADER_BG = (230, 236, 245)
ROW_ALT = (248, 250, 252)
PASS_GREEN = (34, 139, 34)
WARN_ORANGE = (210, 120, 20)


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 10, style="F")
        self.set_y(12)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*NAVY)
        self.cell(95, 6, "US Bank Credit Stress Test", align="L")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(95, 6, "US Bank Credit Stress Test", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(95, 8, "github.com/muhammadumarshinwari/us-bank-credit-stress-test", align="L")
        self.cell(95, 8, f"Page {self.page_no()}", align="R")

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(1.5)

    def section_title(self, title: str) -> None:
        if self.get_y() > 248:
            self.add_page()
        self.ln(4)
        self.set_fill_color(*STEEL)
        self.rect(10, self.get_y(), 3, 8, style="F")
        self.set_x(15)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 8, title)
        self.set_draw_color(*STEEL)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title: str) -> None:
        self.ln(1)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, title)
        self.set_text_color(0, 0, 0)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.2, f"    {text}")

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[float] | None = None,
        status_col: int | None = None,
    ) -> None:
        if not rows:
            return
        widths = col_widths or [190 / len(headers)] * len(headers)
        if self.get_y() > 240:
            self.add_page()
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*HEADER_BG)
        self.set_text_color(*NAVY)
        for i, h in enumerate(headers):
            self.cell(widths[i], 7, h, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(0, 0, 0)
        for r_idx, row in enumerate(rows):
            if self.get_y() > 265:
                self.add_page()
                self.set_x(self.l_margin)
                self.set_font("Helvetica", "B", 8)
                self.set_fill_color(*HEADER_BG)
                self.set_text_color(*NAVY)
                for i, h in enumerate(headers):
                    self.cell(widths[i], 7, h, border=1, fill=True)
                self.ln()
                self.set_font("Helvetica", "", 8)
                self.set_text_color(0, 0, 0)
            self.set_x(self.l_margin)
            alt = r_idx % 2 == 1
            self.set_fill_color(*(ROW_ALT if alt else (255, 255, 255)))
            for i, val in enumerate(row):
                if status_col is not None and i == status_col and val in ("PASS", "WARN", "FAIL"):
                    c = {"PASS": PASS_GREEN, "WARN": WARN_ORANGE, "FAIL": (180, 40, 40)}[val]
                    self.set_fill_color(*c)
                    self.set_text_color(255, 255, 255)
                    self.set_font("Helvetica", "B", 7)
                    self.cell(widths[i], 6, f" {val} ", border=1, fill=True, align="C")
                    self.set_text_color(0, 0, 0)
                    self.set_font("Helvetica", "", 8)
                    self.set_fill_color(*(ROW_ALT if alt else (255, 255, 255)))
                else:
                    self.cell(widths[i], 6, str(val)[:90], border=1, fill=True)
            self.ln()
        self.ln(3)

    def figure(self, path: Path, caption: str, width: float = 182) -> None:
        if not path.exists():
            self.body(f"[Chart missing: {path.name}. Run python examples/run_all.py first.]")
            return
        with Image.open(path) as img:
            h = width * (img.size[1] / img.size[0])
        if self.get_y() + h + 20 > 282:
            self.add_page()
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, caption)
        self.set_text_color(0, 0, 0)
        self.image(str(path), x=(210 - width) / 2, w=width)
        self.ln(h + 5)


def _extra_charts() -> None:
    DOCS.mkdir(exist_ok=True)
    if (DATA / "model_validation_flags.csv").exists():
        flags = pd.read_csv(DATA / "model_validation_flags.csv")
        counts = flags["status"].value_counts().reindex(["PASS", "WARN", "FAIL"]).fillna(0)
        fig, ax = plt.subplots(figsize=(5.5, 3))
        colors = {"PASS": "#228B22", "WARN": "#D27814", "FAIL": "#B42828"}
        bars = ax.bar(counts.index, counts.values, color=[colors[k] for k in counts.index])
        ax.set_title("Validation flags (est. through 2022 Q4)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Checks")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, int(v), ha="center", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(DOCS / "pdf_validation_flags.png", dpi=150, facecolor="white")
        plt.close(fig)


def build_pdf() -> Path:
    _extra_charts()
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(14, 14, 14)

    # ---- Cover ----
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 62, style="F")
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "US Bank Credit Stress Test", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(210, 225, 245)
    pdf.cell(0, 8, "Macro satellite model on public FDIC data", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(22)
    pdf.set_text_color(40, 40, 40)
    pdf.body(
        "A macro stress test on ten large US banks using public FDIC call reports, "
        "FRED macro history, and the 2025 Fed severely adverse scenario. "
        "This report walks through the model, validation, and stress projections."
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.body("MIT License  |  Public data: FDIC, FRED, Federal Reserve scenarios")

    # ---- Section 1 ----
    pdf.add_page()
    pdf.section_title("1. How banks stress credit (and what this project does)")
    pdf.body(
        "In a large bank, credit stress testing usually runs through probability of default (PD), "
        "loss given default (LGD), and exposure at default (EAD). Under IFRS 9, expected credit loss "
        "(ECL) combines those three at the loan or facility level, with ratings migration and "
        "Stage 1, 2, 3 allocation driving provisions. In CCAR and DFAST, banks build segment models "
        "for CRE, cards, mortgages, and C&I, often with collateral and prepayment detail, all "
        "shocked under a supervisory macro scenario."
    )
    pdf.body(
        "That bottom-up stack is the production system. It needs loan tapes, internal ratings, "
        "write-off history, and full model validation every cycle."
    )
    pdf.body(
        "This project is not that. It is a top-down satellite model: NPL ratio regressed on lagged "
        "NPL and a small set of macro variables (GDP, unemployment, mortgage rates). Losses are "
        "backed out with a fixed LGD. Capital is equity over assets with a flat PPNR assumption. "
        "No risk-weighted assets, no AFS marks, no management actions."
    )
    pdf.body(
        "All inputs are public. The workflow follows standard macro stress testing practice: "
        "scenario, satellite model, validation, balance-sheet projection."
    )

    pdf.section_title("2. The workflow")
    pdf.bullet("1. Build a panel: FDIC financials merged with FRED macros (2005 Q1 to 2024 Q4).")
    pdf.bullet("2. Estimate a satellite NPL model with bank fixed effects.")
    pdf.bullet("3. Validate: in-sample backtest, 2023-2024 out-of-sample forecast, diagnostics.")
    pdf.bullet("4. Stress test: 2025 Fed severely adverse scenario to NPL paths, losses, capital.")
    pdf.ln(2)
    pdf.body(
        "Each step is a separate Python module. You can read the scenario loader, regression, "
        "validation flags, and capital formulas independently."
    )

    # ---- Data ----
    pdf.section_title("3. Data")
    pdf.body(
        "Ten banks: JPMorgan, BofA, Wells Fargo, Citi, U.S. Bank, PNC, Truist, Fifth Third, "
        "KeyBank, Regions. Data from the FDIC BankFind API (free, no key). NPL ratio = "
        "NCLNLS / LNLSNET x 100. A PD model would track rating migration. Here I model the "
        "portfolio NPL outcome directly."
    )
    pdf.table(
        ["CERT", "Bank"],
        [
            ["628", "JPMorgan Chase NA"], ["3510", "Bank of America NA"],
            ["3511", "Wells Fargo NA"], ["7213", "Citibank NA"],
            ["6548", "U.S. Bank NA"], ["6384", "PNC Bank NA"],
            ["9846", "Truist Bank"], ["5649", "Fifth Third Bank NA"],
            ["6672", "KeyBank NA"], ["18409", "Regions Bank"],
        ],
        [18, 172],
    )
    pdf.body(
        "780 bank-quarters after lags (10 banks x 80 quarters minus lag rows). "
        "Stress scenario: 2025 Fed severely adverse, nine quarters from 2025 Q1."
    )

    # ---- Model ----
    pdf.add_page()
    pdf.section_title("4. The satellite model")
    pdf.body(
        "NPL(i,t) = a_i + rho*NPL(i,t-1) + b1*GDP(t-2) + b2*UR(t) + b3*Mtg(t-1) + error. "
        "Pooled OLS with bank dummies and HC1 robust standard errors. Bank FE only, no time FE."
    )
    pdf.body(
        "I arrived at four macro terms by dropping lags and variables step by step. GDP at lag 2, "
        "contemporaneous unemployment, mortgage rate at lag 1. HPI, CPI, and T-bill dropped from "
        "the final spec."
    )
    pdf.table(
        ["Variable", "Coef.", "Std err", "Sign", "Reading"],
        [
            ["npl_lag1", "0.942", "0.011", "+", "High persistence"],
            ["real_gdp_growth_lag2", "-0.007", "0.002", "-", "Weaker growth raises NPL"],
            ["unemployment", "0.052", "0.010", "+", "Higher UR raises NPL"],
            ["mortgage_rate_lag1", "0.077", "0.008", "+", "Higher rates raise NPL"],
        ],
        [38, 16, 18, 12, 106],
    )
    pdf.body("R-squared = 0.972. All coefficients significant at 1%. Signs match credit-cycle theory.")

    # ---- Validation ----
    pdf.section_title("5. Model validation")
    pdf.body(
        "The validate_model() function produces PASS, WARN, and FAIL flags. Summary: "
        "12 PASS, 4 WARN, 0 FAIL (estimation through 2022 Q4)."
    )
    if (DOCS / "pdf_validation_flags.png").exists():
        pdf.figure(DOCS / "pdf_validation_flags.png", "Figure 1. Automated validation check counts.")

    flags_path = DATA / "model_validation_flags.csv"
    if flags_path.exists():
        flags = pd.read_csv(flags_path)
        rows = [[r["check"].replace("_", " "), r["status"], r["detail"][:70].replace(";", ",")] for _, r in flags.iterrows()]
        pdf.table(["Check", "Status", "Detail"], rows, [48, 18, 124], status_col=1)

    pdf.body(
        "WARN flags: serial correlation (DW = 1.12), non-normal residuals (GFC fat tails), "
        "unemployment VIF = 10.7, GFC RMSE = 0.75 pp. These are documented, not hidden."
    )

    # ---- Backtest ----
    pdf.add_page()
    pdf.section_title("6. Does the model track realized NPL?")
    pdf.sub_title("In-sample backtest")
    pdf.body(
        "One-step-ahead prediction at each quarter using only prior information. "
        "Overall RMSE = 0.31 pp, MAE = 0.18 pp (percentage points of NPL ratio)."
    )
    pdf.table(
        ["Bank", "RMSE", "MAE"],
        [
            ["Regions", "0.14", "0.11"], ["U.S. Bank", "0.19", "0.13"],
            ["Fifth Third", "0.23", "0.18"], ["Truist", "0.24", "0.16"],
            ["Wells Fargo", "0.33", "0.19"], ["JPMorgan Chase", "0.34", "0.20"],
            ["Citibank", "0.35", "0.21"], ["KeyBank", "0.35", "0.18"],
            ["Bank of America", "0.40", "0.26"], ["PNC", "0.40", "0.18"],
        ],
        [70, 30, 30],
    )
    pdf.table(
        ["Period", "RMSE (pp)"],
        [["2005-2007", "0.20"], ["2008-2009", "0.75"], ["2010-2014", "0.29"],
         ["2015-2019", "0.11"], ["2020-2022", "0.18"]],
        [60, 130],
    )
    pdf.figure(
        DOCS / "backtest_results.png",
        "Figure 2. Top: actual NPL (solid) vs one-step predicted (dashed) by bank. "
        "Bottom: average prediction error by quarter. Errors spike in the GFC.",
    )

    pdf.sub_title("Out-of-sample forecast (2023 Q1 to 2024 Q4)")
    pdf.body(
        "Model estimated through 2022 Q4 only. Holdout: 2023-2024. "
        "RMSE = 0.17 pp, MAE = 0.16 pp, bias = +0.14 pp (model under-predicts realized NPL)."
    )
    pdf.table(
        ["Quarter", "Actual %", "Pred %", "Error pp"],
        [
            ["2023 Q1", "0.86", "0.99", "+0.13"], ["2023 Q2", "0.84", "0.99", "+0.15"],
            ["2023 Q3", "0.91", "0.98", "+0.08"], ["2023 Q4", "0.96", "1.11", "+0.15"],
            ["2024 Q1", "1.01", "1.18", "+0.17"], ["2024 Q2", "0.97", "1.18", "+0.21"],
            ["2024 Q3", "1.02", "1.20", "+0.18"], ["2024 Q4", "1.08", "1.17", "+0.09"],
        ],
        [28, 28, 28, 28],
    )
    pdf.figure(
        DOCS / "forecast_sample_results.png",
        "Figure 3. Out-of-sample window. Lines track more closely than in the GFC. "
        "Bars show the model lagging the 2024 NPL uptick.",
    )

    # ---- Diagnostics ----
    pdf.add_page()
    pdf.sub_title("Regression diagnostics")
    pdf.body(
        "Durbin-Watson = 1.12 (autocorrelation). Jarque-Bera rejects normality (GFC tails). "
        "Breusch-Pagan rejects homoskedasticity (HC1 SEs used). Unemployment VIF = 10.7."
    )
    pdf.figure(
        DOCS / "model_diagnostics.png",
        "Figure 4. Top left: residuals vs fitted. Top right: Q-Q plot (fat upper tail). "
        "Bottom left: mean residual by quarter (GFC spike). Bottom right: RMSE by bank.",
    )

    # ---- Stress ----
    pdf.add_page()
    pdf.section_title("7. Stress test results")
    pdf.body(
        "2025 Fed severely adverse scenario from 2024 Q4 positions. "
        "Losses = delta NPL x net loans x LGD (45%). PPNR = 0.10% quarterly ROA under stress. "
        "Capital = equity / assets. No RWA, no dividends, no management actions."
    )
    pdf.table(
        ["Bank", "Trough cap", "End cap", "Credit loss", "Breach 5%?"],
        [
            ["JPMorgan Chase", "9.10%", "10.04%", "$10.3B", "No"],
            ["Bank of America", "9.58%", "10.46%", "$8.4B", "No"],
            ["U.S. Bank", "9.68%", "10.59%", "$2.0B", "No"],
            ["PNC", "9.71%", "10.56%", "$1.9B", "No"],
            ["Fifth Third", "9.74%", "10.67%", "$0.4B", "No"],
            ["Wells Fargo", "9.84%", "10.67%", "$6.3B", "No"],
            ["Citibank", "10.17%", "11.15%", "$3.7B", "No"],
            ["KeyBank", "10.50%", "11.41%", "$0.6B", "No"],
            ["Truist", "11.73%", "12.61%", "$1.7B", "No"],
            ["Regions", "12.60%", "13.57%", "$0.8B", "No"],
        ],
        [38, 22, 22, 28, 22],
    )
    pdf.figure(
        DOCS / "stress_results.png",
        "Figure 5. Left: NPL roughly doubles under the adverse macro path. "
        "Right: equity/assets ratios. Capital rises slightly because PPNR exceeds losses "
        "in this simplified module. That is a limitation, not a solvency finding.",
        width=178,
    )

    # ---- Close ----
    pdf.section_title("8. Limitations and disclaimer")
    pdf.body(
        "This is not a PD/ECL engine. Capital is not CET1. PPNR is a flat ROA. "
        "The linear model misses GFC nonlinearities. Ten banks is a demo sample."
    )
    pdf.body(
        "Educational project on public data only. Not supervisory software. "
        "Outputs are not assessments of any real bank."
    )
    pdf.body("Repository: github.com/muhammadumarshinwari/us-bank-credit-stress-test")

    DOCS.mkdir(exist_ok=True)
    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF saved to {path}")
