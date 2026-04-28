# Design Spec: Predicting CPI — Macro/Surveys vs. Prediction Markets vs. Sentiment

**Course:** ECON 460 — Machine Learning for Economics, USC Spring 2026  
**Written report due:** May 13, 2026  
**Presentation:** April 28 or 30, 2026 (~15 minutes, work-in-progress is fine)  
**Languages:** Python (data collection/processing) + R (analysis)

---

## Research Question

Which category of information — traditional macro indicators and surveys, prediction market contracts, or text-based sentiment — best predicts monthly U.S. CPI inflation over the period January 2022 to present?

## Thesis

We use a unified LASSO regression where features from all three categories compete simultaneously. The category whose variables retain non-zero coefficients under regularization provides the most independent predictive signal about future CPI. As a baseline, we also run separate OLS regressions per category and compare R² and RMSE across all four models (3 OLS + 1 LASSO).

The central narrative: the 2022–2023 inflation surge was the largest forecasting failure by professional economists in 40 years. Did prediction markets or social media sentiment see it coming better?

---

## Time Period

January 2022 – April 2026 (~52 monthly observations).  
Rationale: Kalshi CPI prediction market contracts only became available around 2022, which constrains the start date. This window also captures the full inflation surge and subsequent disinflation — the most interesting forecasting period.

---

## Outcome Variable

Month `t+1` CPI YoY% (seasonally adjusted, FRED series `CPIAUCSL`).  
All predictors use data available at end of month `t` (shifted by 1 period to prevent look-ahead bias).

---

## Data Sources

### Category 1 — Macro/Surveys (FRED API)
| Variable | FRED Series | Description |
|---|---|---|
| Michigan 1-yr inflation expectations | `MICH` | Consumer survey |
| Cleveland Fed 1-yr inflation expectations | `EXPINF1YR` | Model-based expectations |
| SPF median CPI forecast | Philadelphia Fed API | Professional forecasters |
| Unemployment rate | `UNRATE` | Labor market |
| PPI all commodities | `PPIACO` | Producer prices |
| PCE price index | `PCEPI` | Alternate inflation measure |
| WTI oil price | `DCOILWTICO` | Commodity inflation driver |

### Category 2 — Prediction Markets (Kalshi API)
- Kalshi monthly CPI contracts: derive implied expected CPI point estimate from probability-weighted midpoints of "Will CPI be above X%" binary contracts
- Available from approximately early 2022 onward

### Category 3 — Sentiment (GDELT + Google Trends, both free)
- GDELT: average monthly news tone score on articles mentioning "inflation" or "CPI"
- Google Trends: monthly search volume index for query "inflation" (US)

---

## Methods

### Step 1 — Descriptive Analysis
- Correlation table: all predictors vs. realized CPI YoY
- Time series overlay: each category's key variable vs. actual CPI
- Purpose: motivate the formal analysis, visually identify which category tracked CPI best

### Step 2 — Category-Level OLS (Baseline)
- Three separate OLS regressions, one per category
- Outcome: month `t+1` CPI YoY
- Report: R², RMSE, and coefficient table for each
- Answers: "how good is each category on its own?"

### Step 3 — Unified LASSO (Main Result)
- All features from all three categories in one LASSO regression
- Lambda selected by leave-one-out cross-validation (appropriate for ~52 observations)
- Features retaining non-zero coefficients "win"
- Answers: "which category contains independent information about CPI when all compete?"

### Evaluation
- Summary table: R², RMSE for all four models (3 OLS + 1 LASSO)
- LASSO coefficient path plot (shows how variables enter/exit as lambda varies)
- Discussion: which category survives, economic interpretation of results

---

## Visualizations (1–3 for written report)
1. **Time series plot:** Actual CPI YoY vs. each category's primary predictor (3-panel or overlaid)
2. **Summary table:** R², RMSE across all models — formatted, not raw output
3. **LASSO coefficient plot:** Which variables survive regularization, grouped by category

---

## Code Structure
```
econ460_project/
├── data/
│   ├── raw/          # downloaded CSVs from APIs
│   └── processed/    # cleaned, merged monthly panel
├── python/
│   ├── 01_collect_fred.py       # FRED API: macro + survey data
│   ├── 02_collect_kalshi.py     # Kalshi API: prediction market implied CPI
│   ├── 03_collect_sentiment.py  # GDELT + Google Trends sentiment
│   └── 04_merge_panel.py        # merge all sources into monthly panel
├── R/
│   ├── 01_descriptive.R         # correlation table, time series plots
│   ├── 02_ols_models.R          # category-level OLS regressions
│   ├── 03_lasso_unified.R       # unified LASSO with CV
│   └── 04_tables_figures.R      # formatted output tables and figures
├── output/
│   ├── figures/
│   └── tables/
└── docs/
```

---

## Written Report Structure (~5 pages double-spaced)
1. **Introduction:** Why CPI forecasting matters; the 2022 forecasting failure; our three-way comparison
2. **Dataset:** Sources, time period, variable construction, known limitations (Kalshi data starts 2022, GDELT as news proxy)
3. **Methods:** OLS baseline per category + unified LASSO; leave-one-out CV; look-ahead bias prevention
4. **Results:** Correlation analysis → OLS comparison → LASSO results; which category survives
5. **Conclusion:** What this tells us about information efficiency; limitations; extensions

---

## Known Limitations
- ~52 monthly observations is a small sample; LASSO results should be interpreted cautiously
- Kalshi CPI contracts may have low liquidity in early 2022 — implied probabilities may be noisy
- GDELT captures news tone, not Twitter/Reddit — a proxy for public sentiment rather than direct social media
- SPF forecasts are quarterly; forward-filled to monthly (each quarter's forecast held constant until the next release)
- Results are in-sample comparisons (not out-of-sample forecasts), which is appropriate given the small N and the comparison-focused research question
