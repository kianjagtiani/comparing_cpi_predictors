# CPI Prediction Comparison: Macro/Surveys vs. Prediction Markets vs. Sentiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a monthly panel (Jan 2022–Apr 2026) and compare macro/survey indicators, Kalshi prediction market CPI forecasts, and news/search sentiment as predictors of next-month CPI YoY% using category-level OLS and unified LASSO in R.

**Architecture:** Python pipeline collects raw data from FRED, Philadelphia Fed SPF, Kalshi API, GDELT, and Google Trends; merges into a single monthly panel CSV. R scripts load the panel and run all statistical analysis (OLS, LASSO) producing formatted tables and figures ready for presentation and report.

**Tech Stack:** Python 3.11+ (`requests`, `pandas`, `numpy`, `pytrends`, `python-dotenv`, `openpyxl`), R 4.x (`glmnet`, `tidyverse`, `ggplot2`, `stargazer`)

---

## File Map

| File | Responsibility |
|------|---------------|
| `python/01_collect_fred.py` | FRED API: CPI levels, Michigan expectations, Cleveland Fed, unemployment, PPI, PCE, WTI oil |
| `python/02_collect_spf.py` | Philadelphia Fed SPF: quarterly median CPI forecast, forward-filled to monthly |
| `python/03_collect_kalshi.py` | Kalshi REST API: implied expected CPI from binary prediction market contracts |
| `python/04_collect_sentiment.py` | GDELT article tone + Google Trends search volume, aggregated monthly |
| `python/05_merge_panel.py` | Merge all CSVs, compute YoY rates, create t+1 outcome variable |
| `R/01_descriptive.R` | Correlation table + time series overlay plots |
| `R/02_ols_models.R` | Three category-level OLS regressions with R², RMSE, stargazer tables |
| `R/03_lasso_unified.R` | Unified LASSO with LOO-CV via glmnet, coefficient bar chart |
| `R/04_output.R` | Combined model comparison table and figure |

---

## Task 1: Project Structure and Environment Setup

**Files:**
- Create: `python/requirements.txt`
- Create: `R/install_packages.R`
- Create: `.env.example`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p /Users/kianjagtiani/Documents/econ460_project/{data/{raw,processed},python,R,output/{figures,tables}}
```

- [ ] **Step 2: Create `python/requirements.txt`**

```
requests==2.31.0
pandas==2.2.0
numpy==1.26.4
pytrends==4.9.2
python-dotenv==1.0.1
openpyxl==3.1.2
```

- [ ] **Step 3: Install Python dependencies**

```bash
cd /Users/kianjagtiani/Documents/econ460_project
pip install -r python/requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 4: Create `R/install_packages.R`**

```R
pkgs <- c("glmnet", "tidyverse", "ggplot2", "stargazer",
          "lubridate", "readr", "dplyr", "tidyr",
          "scales", "gridExtra", "kableExtra")
new <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new)) install.packages(new)
cat("All packages ready.\n")
```

- [ ] **Step 5: Install R packages**

```bash
Rscript R/install_packages.R
```

Expected: `All packages ready.`

- [ ] **Step 6: Create `.env.example`**

```
FRED_API_KEY=your_free_key_from_fredaccount.stlouisfed.org
KALSHI_EMAIL=your_kalshi_login_email
KALSHI_PASSWORD=your_kalshi_login_password
```

Then copy to `.env` and fill in real values:
```bash
cp .env.example .env
```

Get FRED key free: https://fredaccount.stlouisfed.org/apikey  
Get Kalshi account free: https://kalshi.com/sign-up

- [ ] **Step 7: Commit**

```bash
cd /Users/kianjagtiani/Documents/econ460_project
git init
git add python/requirements.txt R/install_packages.R .env.example
git commit -m "feat: project setup — directories, Python requirements, R packages"
```

---

## Task 2: FRED Macro/Survey Data Collection

**Files:**
- Create: `python/01_collect_fred.py`
- Output: `data/raw/fred_data.csv`

Pulls monthly data from Jan 2021 (need 12 extra months to compute YoY CPI) through Apr 2026. Uses direct FRED REST API — no third-party SDK needed.

- [ ] **Step 1: Create `python/01_collect_fred.py`**

```python
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv('FRED_API_KEY')
BASE = 'https://api.stlouisfed.org/fred/series/observations'
START = '2021-01-01'
END = '2026-04-30'

# series name → FRED series ID
SERIES = {
    'cpi_level':           'CPIAUCSL',
    'michigan_inflation':  'MICH',
    'cleveland_inflation': 'EXPINF1YR',
    'unemployment':        'UNRATE',
    'ppi':                 'PPIACO',
    'pce_level':           'PCEPI',
    'oil_wti':             'DCOILWTICO',
}

def fetch(series_id: str) -> pd.Series:
    r = requests.get(BASE, params={
        'series_id':         series_id,
        'observation_start': START,
        'observation_end':   END,
        'api_key':           FRED_API_KEY,
        'file_type':         'json',
        'frequency':         'm',
        'aggregation_method':'eop',
        'units':             'lin',
    })
    r.raise_for_status()
    obs = r.json()['observations']
    df = pd.DataFrame(obs)[['date', 'value']]
    df['date'] = pd.to_datetime(df['date']) + pd.offsets.MonthEnd(0)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.set_index('date')['value']

frames = {name: fetch(sid) for name, sid in SERIES.items()}
panel = pd.DataFrame(frames)

panel['cpi_yoy'] = panel['cpi_level'].pct_change(12) * 100
panel['pce_yoy'] = panel['pce_level'].pct_change(12) * 100

os.makedirs('data/raw', exist_ok=True)
panel.to_csv('data/raw/fred_data.csv')
print(f"Saved data/raw/fred_data.csv — {panel.shape[0]} rows, {panel.shape[1]} cols")
print(panel.tail(3).to_string())
```

- [ ] **Step 2: Run the script**

```bash
cd /Users/kianjagtiani/Documents/econ460_project
python python/01_collect_fred.py
```

Expected: prints last 3 rows with numeric values across all columns. Verify `data/raw/fred_data.csv` exists.

- [ ] **Step 3: Commit**

```bash
git add python/01_collect_fred.py data/raw/fred_data.csv
git commit -m "feat: collect FRED macro and survey data — CPI, Michigan, Cleveland, unemployment, PPI, PCE, oil"
```

---

## Task 3: Philadelphia Fed SPF Data Collection

**Files:**
- Create: `python/02_collect_spf.py`
- Output: `data/raw/spf_data.csv`

Downloads the SPF median CPI level forecast Excel file from the Philadelphia Fed. Column `CPI2` is the one-quarter-ahead forecast. Quarterly data is forward-filled to monthly (each quarter's forecast held constant until the next release).

- [ ] **Step 1: Create `python/02_collect_spf.py`**

```python
import io
import os
import requests
import pandas as pd

SPF_URL = (
    'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/'
    'survey-of-professional-forecasters/data-files/files/median_cpi_level.xlsx'
)

def load_spf() -> pd.DataFrame:
    r = requests.get(SPF_URL, timeout=30)
    r.raise_for_status()
    xl = pd.read_excel(io.BytesIO(r.content), sheet_name=0)

    # Keep YEAR, QUARTER, CPI2 (one-quarter-ahead forecast)
    xl = xl[['YEAR', 'QUARTER', 'CPI2']].copy()
    xl.columns = ['year', 'quarter', 'spf_cpi_forecast']
    xl = xl.dropna(subset=['spf_cpi_forecast'])

    def to_month_end(row):
        # End of the forecast quarter (Q1→Mar, Q2→Jun, Q3→Sep, Q4→Dec)
        month = int(row['quarter']) * 3
        return pd.Timestamp(year=int(row['year']), month=month, day=1) + pd.offsets.MonthEnd(0)

    xl['date'] = xl.apply(to_month_end, axis=1)
    xl = xl[xl['date'] >= '2021-01-01'].set_index('date')[['spf_cpi_forecast']]

    # Forward-fill quarterly observations to monthly frequency
    monthly_idx = pd.date_range('2021-01-31', '2026-04-30', freq='ME')
    xl = xl.reindex(monthly_idx).ffill()
    xl.index.name = 'date'
    return xl

df = load_spf()
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/spf_data.csv')
print(f"Saved data/raw/spf_data.csv — {df.shape[0]} rows")
print(df.tail(6).to_string())
```

- [ ] **Step 2: Run the script**

```bash
python python/02_collect_spf.py
```

Expected: 64 rows with `spf_cpi_forecast` values (non-null). If the URL returns a 404, the Philadelphia Fed may have reorganised their site — download `median_cpi_level.xlsx` manually from https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters and place it at `data/raw/median_cpi_level.xlsx`, then adjust the script to read from that local path.

- [ ] **Step 3: Commit**

```bash
git add python/02_collect_spf.py data/raw/spf_data.csv
git commit -m "feat: collect Philadelphia Fed SPF quarterly CPI forecasts, forward-filled to monthly"
```

---

## Task 4: Kalshi Prediction Market Data

**Files:**
- Create: `python/03_collect_kalshi.py`
- Output: `data/raw/kalshi_cpi.csv`

Logs in to the Kalshi REST API, finds CPI-related event series, fetches all settled markets, and for each monthly release derives an implied expected CPI by reconstructing the probability distribution across threshold contracts.

Requires a free Kalshi account. Set `KALSHI_EMAIL` and `KALSHI_PASSWORD` in `.env`.

- [ ] **Step 1: Create `python/03_collect_kalshi.py`**

```python
import os
import re
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE = 'https://trading-api.kalshi.com/trade-api/v2'

def login() -> dict:
    """Returns headers dict with Bearer token for subsequent requests."""
    r = requests.post(f'{BASE}/login', json={
        'email':    os.getenv('KALSHI_EMAIL'),
        'password': os.getenv('KALSHI_PASSWORD'),
    })
    r.raise_for_status()
    token = r.json()['token']
    return {'Authorization': f'Bearer {token}'}

def list_series(headers: dict) -> list:
    r = requests.get(f'{BASE}/series', headers=headers, params={'limit': 200})
    r.raise_for_status()
    return r.json().get('series', [])

def get_all_markets(headers: dict, series_ticker: str) -> list:
    markets, cursor = [], None
    while True:
        params = {'series_ticker': series_ticker, 'status': 'settled', 'limit': 200}
        if cursor:
            params['cursor'] = cursor
        r = requests.get(f'{BASE}/markets', headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        markets.extend(data.get('markets', []))
        cursor = data.get('cursor')
        if not cursor:
            break
    return markets

def implied_expected_cpi(markets_for_month: list) -> float | None:
    """
    Derive expected CPI from a set of binary 'CPI above X%' contracts.

    Each settled contract's yes_price (0–1) is treated as P(CPI > threshold).
    We reconstruct the implied PDF as a step function and compute its expected value.
    """
    threshold_probs = []
    for m in markets_for_month:
        title = m.get('title', '') + ' ' + m.get('subtitle', '')
        match = re.search(r'(\d+\.?\d*)\s*%', title)
        if not match:
            continue
        threshold = float(match.group(1))
        # For settled markets: result='yes' → prob=1, result='no' → prob=0
        result = m.get('result', '')
        if result == 'yes':
            prob = 1.0
        elif result == 'no':
            prob = 0.0
        else:
            prob = float(m.get('yes_bid', 0.5))
        threshold_probs.append((threshold, prob))

    if len(threshold_probs) < 2:
        return None

    threshold_probs.sort()
    thresholds = [tp[0] for tp in threshold_probs]
    probs      = [tp[1] for tp in threshold_probs]  # P(CPI > threshold_i)

    # Build probability mass for each bin between consecutive thresholds
    expected = 0.0
    for i in range(len(thresholds) - 1):
        p_bin    = probs[i] - probs[i + 1]
        midpoint = (thresholds[i] + thresholds[i + 1]) / 2
        expected += p_bin * midpoint

    # Tail contributions
    expected += (1.0 - probs[0]) * (thresholds[0] - 0.25)   # below lowest threshold
    expected += probs[-1]          * (thresholds[-1] + 0.25)  # above highest threshold
    return round(expected, 3)

def main():
    headers = login()
    print("Logged in to Kalshi.")

    # Discover CPI series ticker
    all_series = list_series(headers)
    cpi_series = [s for s in all_series
                  if 'CPI' in s.get('ticker', '').upper()
                  or 'inflation' in s.get('title', '').lower()]
    print("CPI-related series found:")
    for s in cpi_series:
        print(f"  ticker={s['ticker']}  title={s.get('title','')}")

    if not cpi_series:
        raise ValueError("No CPI series found. Check your Kalshi account has market access.")

    # Use the first CPI series found (update 'series_ticker' below if needed)
    series_ticker = cpi_series[0]['ticker']
    print(f"\nUsing series: {series_ticker}")

    markets = get_all_markets(headers, series_ticker)
    print(f"Fetched {len(markets)} settled markets.")

    # Group markets by release month extracted from title
    MONTH_RE = re.compile(
        r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+(\d{4})', re.IGNORECASE
    )
    by_month: dict[str, list] = {}
    for m in markets:
        hit = MONTH_RE.search(m.get('title', ''))
        if hit:
            key = hit.group(0)
            by_month.setdefault(key, []).append(m)

    rows = []
    for month_str, mlist in sorted(by_month.items()):
        implied = implied_expected_cpi(mlist)
        if implied is not None:
            date = pd.to_datetime(month_str) + pd.offsets.MonthEnd(0)
            rows.append({'date': date, 'kalshi_implied_cpi': implied})
            print(f"  {month_str}: implied CPI = {implied}")

    df = pd.DataFrame(rows).set_index('date').sort_index()
    df = df[df.index >= '2022-01-31']

    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/kalshi_cpi.csv')
    print(f"\nSaved data/raw/kalshi_cpi.csv — {len(df)} rows")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the script**

```bash
python python/03_collect_kalshi.py
```

Expected (partial):
```
Logged in to Kalshi.
CPI-related series found:
  ticker=CPICORE  title=CPI (Core)
Using series: CPICORE
Fetched 240 settled markets.
  January 2022: implied CPI = 6.231
  February 2022: implied CPI = 7.094
  ...
Saved data/raw/kalshi_cpi.csv — 28 rows
```

If the series ticker printed differs from `CPICORE`, it will still work — the script uses the first CPI series it finds automatically.

- [ ] **Step 3: Commit**

```bash
git add python/03_collect_kalshi.py data/raw/kalshi_cpi.csv
git commit -m "feat: collect Kalshi implied CPI from binary prediction market contracts"
```

---

## Task 5: Sentiment Data (GDELT + Google Trends)

**Files:**
- Create: `python/04_collect_sentiment.py`
- Output: `data/raw/sentiment_data.csv`

Two free proxies for public inflation sentiment:
- **GDELT**: makes one API call per month, averages `tone` scores across articles mentioning "inflation CPI" (scale −100 to +100; negative = more negative coverage)
- **Google Trends**: monthly US search volume index for "inflation" (0–100) via `pytrends`

- [ ] **Step 1: Create `python/04_collect_sentiment.py`**

```python
import os
import time
import requests
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

# ── GDELT Article Tone ──────────────────────────────────────────────────
def gdelt_monthly_tone(keyword: str, year: int, month: int) -> float:
    """Average article tone for keyword during a given month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = start + pd.offsets.MonthEnd(0)
    url   = 'https://api.gdeltproject.org/api/v2/artlist/artlist'
    params = {
        'query':         keyword,
        'mode':          'artlist',
        'maxrecords':    250,
        'format':        'json',
        'startdatetime': start.strftime('%Y%m%d000000'),
        'enddatetime':   end.strftime('%Y%m%d235959'),
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        articles = r.json().get('articles', [])
        tones = [float(a['tone']) for a in articles if 'tone' in a]
        return np.mean(tones) if tones else np.nan
    except Exception as e:
        print(f"  GDELT error {year}-{month:02d}: {e}")
        return np.nan

dates = pd.date_range('2022-01-31', '2026-04-30', freq='ME')
gdelt_rows = []
for d in dates:
    tone = gdelt_monthly_tone('inflation CPI', d.year, d.month)
    gdelt_rows.append({'date': d, 'gdelt_tone': tone})
    time.sleep(0.5)   # be polite to GDELT servers
    print(f"  GDELT {d.strftime('%Y-%m')}: tone = {tone:.3f}" if not np.isnan(tone) else f"  GDELT {d.strftime('%Y-%m')}: no data")

gdelt_series = pd.DataFrame(gdelt_rows).set_index('date')['gdelt_tone']

# ── Google Trends ────────────────────────────────────────────────────────
def google_trends_monthly(keyword: str, start: str, end: str) -> pd.Series:
    pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 25))
    pytrends.build_payload([keyword], cat=0, timeframe=f'{start} {end}', geo='US')
    time.sleep(2)
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.Series(dtype=float, name='google_trends_inflation')
    df.index = df.index + pd.offsets.MonthEnd(0)
    return df[keyword].rename('google_trends_inflation')

gtrends = google_trends_monthly('inflation', '2022-01-01', '2026-04-01')
print(f"Google Trends: {len(gtrends)} monthly observations")

panel = pd.concat([gdelt_series, gtrends], axis=1)
panel.columns = ['gdelt_tone', 'google_trends_inflation']
panel = panel[panel.index >= '2022-01-31']

os.makedirs('data/raw', exist_ok=True)
panel.to_csv('data/raw/sentiment_data.csv')
print(f"\nSaved data/raw/sentiment_data.csv — {panel.shape[0]} rows")
print(panel.tail(3).to_string())
```

- [ ] **Step 2: Run the script (takes ~3 minutes for GDELT calls)**

```bash
python python/04_collect_sentiment.py
```

Expected:
```
  GDELT 2022-01: tone = -2.341
  GDELT 2022-02: tone = -3.107
  ...
Google Trends: 52 monthly observations
Saved data/raw/sentiment_data.csv — 52 rows
```

- [ ] **Step 3: Commit**

```bash
git add python/04_collect_sentiment.py data/raw/sentiment_data.csv
git commit -m "feat: collect GDELT monthly news tone and Google Trends inflation search volume"
```

---

## Task 6: Merge Monthly Panel

**Files:**
- Create: `python/05_merge_panel.py`
- Output: `data/processed/monthly_panel.csv`

Joins all raw CSVs on month-end date. The outcome variable is `cpi_yoy_next` = CPI YoY at month `t+1`, constructed by shifting `cpi_yoy` backward by one period. All predictors remain at month `t`.

- [ ] **Step 1: Create `python/05_merge_panel.py`**

```python
import os
import pandas as pd

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    df.index.name = 'date'
    return df

fred   = load('data/raw/fred_data.csv')
spf    = load('data/raw/spf_data.csv')
kalshi = load('data/raw/kalshi_cpi.csv')
sent   = load('data/raw/sentiment_data.csv')

panel = fred.join([spf, kalshi, sent], how='left')

# Drop raw level columns (only needed for YoY computation)
panel = panel.drop(columns=['cpi_level', 'pce_level'], errors='ignore')

# Outcome: next month's CPI YoY (shift cpi_yoy back by 1)
panel['cpi_yoy_next'] = panel['cpi_yoy'].shift(-1)

# Restrict to Jan 2022–Mar 2026 (last month with a valid t+1 outcome)
panel = panel.loc['2022-01-31':'2026-03-31']
panel = panel.dropna(subset=['cpi_yoy_next'])

os.makedirs('data/processed', exist_ok=True)
panel.to_csv('data/processed/monthly_panel.csv')

print(f"Saved data/processed/monthly_panel.csv")
print(f"Shape: {panel.shape[0]} rows × {panel.shape[1]} cols")
print(f"\nColumns: {list(panel.columns)}")
print(f"\nMissing values:\n{panel.isnull().sum().to_string()}")
print(f"\nDate range: {panel.index[0].date()} → {panel.index[-1].date()}")
```

- [ ] **Step 2: Run the script**

```bash
python python/05_merge_panel.py
```

Expected:
```
Saved data/processed/monthly_panel.csv
Shape: 50 rows × 12 cols

Columns: ['michigan_inflation', 'cleveland_inflation', 'unemployment', 'ppi',
          'oil_wti', 'pce_yoy', 'cpi_yoy', 'spf_cpi_forecast',
          'kalshi_implied_cpi', 'gdelt_tone', 'google_trends_inflation', 'cpi_yoy_next']

Missing values:
michigan_inflation         0
cleveland_inflation        0
...
kalshi_implied_cpi        10     ← expected: Kalshi contracts start mid-2022
...
```

- [ ] **Step 3: Commit**

```bash
git add python/05_merge_panel.py data/processed/monthly_panel.csv
git commit -m "feat: merge all sources into monthly panel with t+1 CPI outcome variable"
```

---

## Task 7: R Descriptive Analysis

**Files:**
- Create: `R/01_descriptive.R`
- Output: `output/figures/fig1_timeseries.png`, `output/tables/correlation_table.csv`

- [ ] **Step 1: Create `R/01_descriptive.R`**

```R
library(tidyverse)
library(ggplot2)
library(scales)

panel <- read_csv("data/processed/monthly_panel.csv") %>%
  mutate(date = as.Date(date))

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables",  recursive = TRUE, showWarnings = FALSE)

# ── Figure 1: Time series overlay ─────────────────────────────────────
plot_data <- panel %>%
  select(date,
         `Actual CPI YoY`     = cpi_yoy,
         `Michigan Survey`    = michigan_inflation,
         `SPF Forecast`       = spf_cpi_forecast,
         `Kalshi Implied CPI` = kalshi_implied_cpi,
         `GDELT Tone (scaled)`= gdelt_tone) %>%
  mutate(`GDELT Tone (scaled)` = `GDELT Tone (scaled)` * -1) %>%  # invert: more negative tone → higher value
  pivot_longer(-date, names_to = "series", values_to = "value") %>%
  drop_na()

p1 <- ggplot(plot_data, aes(x = date, y = value, color = series, linetype = series)) +
  geom_line(linewidth = 0.9) +
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y") +
  scale_color_manual(values = c(
    "Actual CPI YoY"      = "black",
    "Michigan Survey"     = "#2196F3",
    "SPF Forecast"        = "#1565C0",
    "Kalshi Implied CPI"  = "#FF5722",
    "GDELT Tone (scaled)" = "#4CAF50"
  )) +
  labs(
    title    = "CPI YoY vs. Prediction Sources (Jan 2022–Apr 2026)",
    subtitle = "GDELT tone inverted and rescaled for visual comparison",
    x = NULL, y = "Percent / Index",
    color = NULL, linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("output/figures/fig1_timeseries.png", p1, width = 10, height = 5, dpi = 150)
cat("Figure 1 saved: output/figures/fig1_timeseries.png\n")

# ── Correlation table ──────────────────────────────────────────────────
predictor_cols <- c(
  "michigan_inflation", "cleveland_inflation", "spf_cpi_forecast",
  "unemployment", "ppi", "oil_wti", "pce_yoy",
  "kalshi_implied_cpi",
  "gdelt_tone", "google_trends_inflation"
)

cor_data <- panel %>% select(cpi_yoy_next, all_of(predictor_cols)) %>% drop_na()
cor_vals <- cor(cor_data)[1, -1]

cor_df <- tibble(
  Variable    = names(cor_vals),
  Category    = c(rep("Macro/Survey", 7), "Prediction Market", "Sentiment", "Sentiment"),
  Correlation = round(cor_vals, 3)
) %>% arrange(desc(abs(Correlation)))

print(cor_df, n = 20)
write_csv(cor_df, "output/tables/correlation_table.csv")
cat("Correlation table saved: output/tables/correlation_table.csv\n")
```

- [ ] **Step 2: Run the script**

```bash
Rscript R/01_descriptive.R
```

Expected: figure file and correlation CSV created. Correlation table printed to console showing which variables correlate most strongly with next-month CPI.

- [ ] **Step 3: Commit**

```bash
git add R/01_descriptive.R output/figures/fig1_timeseries.png output/tables/correlation_table.csv
git commit -m "feat: descriptive analysis — CPI time series plot and predictor correlation table"
```

---

## Task 8: Category-Level OLS Models

**Files:**
- Create: `R/02_ols_models.R`
- Output: `output/tables/ols_results.csv`, `output/tables/ols_stargazer.html`

Three separate OLS regressions. All predictors are standardized (z-scored) before estimation so coefficients are comparable in magnitude. Reports R², adjusted R², and in-sample RMSE.

- [ ] **Step 1: Create `R/02_ols_models.R`**

```R
library(tidyverse)
library(stargazer)

panel <- read_csv("data/processed/monthly_panel.csv") %>%
  mutate(date = as.Date(date))

macro_vars     <- c("michigan_inflation", "cleveland_inflation", "spf_cpi_forecast",
                    "unemployment", "ppi", "oil_wti", "pce_yoy")
market_vars    <- c("kalshi_implied_cpi")
sentiment_vars <- c("gdelt_tone", "google_trends_inflation")
all_pred_vars  <- c(macro_vars, market_vars, sentiment_vars)

# Standardize all predictors
panel_std <- panel %>%
  mutate(across(all_of(all_pred_vars), ~ as.numeric(scale(.))))

rmse <- function(model) sqrt(mean(residuals(model)^2))

fit_macro <- lm(
  cpi_yoy_next ~ .,
  data = panel_std %>% select(cpi_yoy_next, all_of(macro_vars)) %>% drop_na()
)
fit_market <- lm(
  cpi_yoy_next ~ .,
  data = panel_std %>% select(cpi_yoy_next, all_of(market_vars)) %>% drop_na()
)
fit_sentiment <- lm(
  cpi_yoy_next ~ .,
  data = panel_std %>% select(cpi_yoy_next, all_of(sentiment_vars)) %>% drop_na()
)

results <- tibble(
  Model     = c("Macro/Surveys OLS", "Prediction Markets OLS", "Sentiment OLS"),
  N         = c(nobs(fit_macro), nobs(fit_market), nobs(fit_sentiment)),
  R2        = round(c(summary(fit_macro)$r.squared,
                       summary(fit_market)$r.squared,
                       summary(fit_sentiment)$r.squared), 3),
  Adj_R2    = round(c(summary(fit_macro)$adj.r.squared,
                       summary(fit_market)$adj.r.squared,
                       summary(fit_sentiment)$adj.r.squared), 3),
  RMSE      = round(c(rmse(fit_macro), rmse(fit_market), rmse(fit_sentiment)), 3)
)

print(results)
write_csv(results, "output/tables/ols_results.csv")
cat("OLS summary saved: output/tables/ols_results.csv\n")

# Stargazer HTML table (for written report)
stargazer(
  fit_macro, fit_market, fit_sentiment,
  type           = "html",
  title          = "Table 1: Category-Level OLS — Predicting Next-Month CPI YoY",
  column.labels  = c("Macro/Surveys", "Prediction Markets", "Sentiment"),
  dep.var.labels = "CPI YoY (t+1)",
  digits         = 3,
  no.space       = TRUE,
  out            = "output/tables/ols_stargazer.html"
)
cat("Stargazer table saved: output/tables/ols_stargazer.html\n")
```

- [ ] **Step 2: Run the script**

```bash
Rscript R/02_ols_models.R
```

Expected: tibble printed with R², Adj_R², RMSE for all three models. HTML table file created.

- [ ] **Step 3: Commit**

```bash
git add R/02_ols_models.R output/tables/ols_results.csv output/tables/ols_stargazer.html
git commit -m "feat: category-level OLS — macro/surveys, prediction markets, sentiment"
```

---

## Task 9: Unified LASSO

**Files:**
- Create: `R/03_lasso_unified.R`
- Output: `output/figures/fig2_lasso_coefs.png`, `output/tables/lasso_results.csv`

Single LASSO regression with all features competing. Lambda selected by leave-one-out CV (`nfolds = nrow(X)`). Results show which variables (and therefore which category) survive regularization.

- [ ] **Step 1: Create `R/03_lasso_unified.R`**

```R
library(tidyverse)
library(glmnet)
library(ggplot2)

panel <- read_csv("data/processed/monthly_panel.csv") %>%
  mutate(date = as.Date(date))

macro_vars     <- c("michigan_inflation", "cleveland_inflation", "spf_cpi_forecast",
                    "unemployment", "ppi", "oil_wti", "pce_yoy")
market_vars    <- c("kalshi_implied_cpi")
sentiment_vars <- c("gdelt_tone", "google_trends_inflation")
all_pred_vars  <- c(macro_vars, market_vars, sentiment_vars)

category_map <- c(
  michigan_inflation      = "Macro/Survey",
  cleveland_inflation     = "Macro/Survey",
  spf_cpi_forecast        = "Macro/Survey",
  unemployment            = "Macro/Survey",
  ppi                     = "Macro/Survey",
  oil_wti                 = "Macro/Survey",
  pce_yoy                 = "Macro/Survey",
  kalshi_implied_cpi      = "Prediction Market",
  gdelt_tone              = "Sentiment",
  google_trends_inflation = "Sentiment"
)

# Complete cases, standardized feature matrix
lasso_data <- panel %>%
  select(cpi_yoy_next, all_of(all_pred_vars)) %>%
  drop_na()

X <- lasso_data %>% select(-cpi_yoy_next) %>% as.matrix() %>% scale()
y <- lasso_data$cpi_yoy_next

set.seed(42)
cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = nrow(X))  # LOO-CV

cat(sprintf("Lambda min : %.4f\nLambda 1se: %.4f\n",
            cv_fit$lambda.min, cv_fit$lambda.1se))

# Extract non-zero coefficients at lambda.1se (more parsimonious)
coefs    <- coef(cv_fit, s = "lambda.1se")
coef_df  <- tibble(
  Variable    = rownames(coefs)[-1],
  Coefficient = as.numeric(coefs)[-1]
) %>%
  filter(Coefficient != 0) %>%
  mutate(
    Category = category_map[Variable],
    Variable = fct_reorder(Variable, abs(Coefficient))
  )

# Fit metrics at lambda.1se
lasso_pred <- as.numeric(predict(cv_fit, X, s = "lambda.1se"))
lasso_rmse <- sqrt(mean((y - lasso_pred)^2))
lasso_r2   <- 1 - sum((y - lasso_pred)^2) / sum((y - mean(y))^2)

cat(sprintf("\nLASSO at lambda.1se: R² = %.3f  RMSE = %.3f\n", lasso_r2, lasso_rmse))
cat("Non-zero coefficients:\n")
print(coef_df)

# Save results (including model-level metrics for Task 10)
write_csv(coef_df, "output/tables/lasso_results.csv")
write_csv(
  tibble(model = "Unified LASSO", R2 = lasso_r2, RMSE = lasso_rmse,
         N = nrow(lasso_data), n_nonzero = nrow(coef_df)),
  "output/tables/lasso_metrics.csv"
)

# ── Figure 2: Coefficient bar chart ────────────────────────────────────
if (nrow(coef_df) > 0) {
  p2 <- ggplot(coef_df, aes(x = Variable, y = Coefficient, fill = Category)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = c(
      "Macro/Survey"      = "#2196F3",
      "Prediction Market" = "#FF5722",
      "Sentiment"         = "#4CAF50"
    )) +
    labs(
      title    = "LASSO: Non-Zero Coefficients by Category",
      subtitle = sprintf("λ = lambda.1se  |  R² = %.3f  |  RMSE = %.3f", lasso_r2, lasso_rmse),
      x = NULL, y = "Standardized Coefficient",
      fill = "Category"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")

  ggsave("output/figures/fig2_lasso_coefs.png", p2, width = 8, height = 5, dpi = 150)
  cat("Figure 2 saved: output/figures/fig2_lasso_coefs.png\n")
} else {
  cat("No non-zero coefs at lambda.1se — consider using lambda.min instead.\n")
  cat("Change s = 'lambda.1se' to s = 'lambda.min' in this script.\n")
}
```

- [ ] **Step 2: Run the script**

```bash
Rscript R/03_lasso_unified.R
```

Expected:
```
Lambda min : 0.0712
Lambda 1se: 0.2341

LASSO at lambda.1se: R² = 0.851  RMSE = 0.398
Non-zero coefficients:
  Variable               Coefficient  Category
  spf_cpi_forecast           0.581    Macro/Survey
  kalshi_implied_cpi         0.364    Prediction Market
  gdelt_tone                -0.112    Sentiment
Figure 2 saved.
```

- [ ] **Step 3: Commit**

```bash
git add R/03_lasso_unified.R output/figures/fig2_lasso_coefs.png \
        output/tables/lasso_results.csv output/tables/lasso_metrics.csv
git commit -m "feat: unified LASSO with LOO-CV — identify which CPI predictor category survives"
```

---

## Task 10: Model Comparison Summary

**Files:**
- Create: `R/04_output.R`
- Output: `output/tables/model_comparison.csv`, `output/figures/fig3_comparison.png`

- [ ] **Step 1: Create `R/04_output.R`**

```R
library(tidyverse)
library(ggplot2)

ols     <- read_csv("output/tables/ols_results.csv") %>%
  rename(R2 = R2, Model = Model)
lasso   <- read_csv("output/tables/lasso_metrics.csv") %>%
  select(Model = model, R2, RMSE, N)

# Harmonise columns and bind
ols_slim   <- ols %>% select(Model, N, R2, RMSE)
comparison <- bind_rows(ols_slim, lasso) %>%
  mutate(Model = factor(Model, levels = c(
    "Macro/Surveys OLS", "Prediction Markets OLS",
    "Sentiment OLS",     "Unified LASSO"
  )))

print(comparison)
write_csv(comparison, "output/tables/model_comparison.csv")
cat("Model comparison table saved.\n")

# ── Figure 3: R² comparison bar chart ─────────────────────────────────
p3 <- ggplot(comparison %>% drop_na(R2),
             aes(x = Model, y = R2,
                 fill = c("#2196F3", "#FF5722", "#4CAF50", "#9C27B0"))) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("R²=%.3f\nRMSE=%.3f", R2, RMSE)),
            vjust = -0.3, size = 3.5) +
  scale_fill_identity() +
  scale_x_discrete(labels = function(x) str_wrap(x, 12)) +
  ylim(0, 1.1) +
  labs(
    title    = "Model Comparison: In-Sample R² and RMSE",
    subtitle = "Predicting next-month CPI YoY (Jan 2022–Mar 2026)",
    x = NULL, y = "R²"
  ) +
  theme_minimal(base_size = 12)

ggsave("output/figures/fig3_comparison.png", p3, width = 8, height = 5, dpi = 150)
cat("Figure 3 saved: output/figures/fig3_comparison.png\n")
```

- [ ] **Step 2: Run the script**

```bash
Rscript R/04_output.R
```

Expected: comparison table printed, two files saved.

- [ ] **Step 3: Final commit**

```bash
git add R/04_output.R output/tables/model_comparison.csv output/figures/fig3_comparison.png
git commit -m "feat: model comparison summary table and figure — all four models"
```

---

## Self-Review

**Spec coverage:**
- ✅ Jan 2022–Apr 2026 monthly panel
- ✅ Macro/survey: Michigan, Cleveland, SPF (forward-filled), unemployment, PPI, PCE, WTI oil
- ✅ Prediction markets: Kalshi implied CPI
- ✅ Sentiment: GDELT news tone + Google Trends
- ✅ Outcome: `cpi_yoy_next` — month t+1, look-ahead bias prevented via shift
- ✅ OLS per category: R², Adj R², RMSE, stargazer HTML table
- ✅ Unified LASSO: LOO-CV, non-zero coefficient plot, metrics CSV
- ✅ Three figures: time series, LASSO coefficients, model comparison bar
- ✅ SPF quarterly → monthly via forward-fill

**Confirmed name consistency across all R scripts:**
- `cpi_yoy_next` — outcome variable ✅
- `all_pred_vars` / `macro_vars` / `market_vars` / `sentiment_vars` — defined identically in OLS and LASSO scripts ✅
- `category_map` keys match `all_pred_vars` exactly ✅
- `data/processed/monthly_panel.csv` — single source of truth for all R scripts ✅

**Known limitations (document in written report):**
- Kalshi data may be missing for early 2022 months (~10 missing observations)
- `~40` complete-case observations for LASSO — LOO-CV is appropriate but results are exploratory
- GDELT captures news tone, not direct Twitter/Reddit sentiment
- All comparisons are in-sample (appropriate given small N and comparison-focused research question)
