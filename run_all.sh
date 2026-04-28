#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== CPI Prediction Pipeline ==="
echo ""

echo "[1/5] Collecting FRED macro/survey data..."
python python/01_collect_fred.py

echo ""
echo "[2/5] Collecting Philadelphia Fed SPF forecasts..."
python python/02_collect_spf.py

echo ""
echo "[3/5] Collecting Kalshi prediction market data..."
python python/03_collect_kalshi.py

echo ""
echo "[4/5] Collecting sentiment data (GDELT + Google Trends, ~3 min)..."
python python/04_collect_sentiment.py

echo ""
echo "[5/5] Merging into monthly panel..."
python python/05_merge_panel.py

echo ""
echo "=== Running R Analysis ==="
echo ""

echo "[R 1/4] Descriptive analysis..."
Rscript R/01_descriptive.R

echo ""
echo "[R 2/4] Category-level OLS models..."
Rscript R/02_ols_models.R

echo ""
echo "[R 3/4] Unified LASSO..."
Rscript R/03_lasso_unified.R

echo ""
echo "[R 4/4] Model comparison output..."
Rscript R/04_output.R

echo ""
echo "=== Pipeline complete ==="
echo "Figures: output/figures/"
echo "Tables:  output/tables/"
