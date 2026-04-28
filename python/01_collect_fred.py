"""
Collect FRED macro series via the public CSV download endpoint.
No API key required.

Uses the macOS system curl binary to bypass LibreSSL TLS limitations
in the Python 3.9 system interpreter (LibreSSL 2.8.3 is incompatible
with urllib3 v2 HTTPS). curl uses macOS SecureTransport and works correctly.

Daily series (e.g. DCOILWTICO) are resampled to monthly end-of-period
last observation before joining to avoid index duplication.
"""
import os
import subprocess
import pandas as pd
from io import StringIO

START = '2021-01-01'
END   = '2026-04-30'

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
    """Download a FRED series from the public CSV endpoint as a monthly pd.Series."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    cmd = f'curl -s --max-time 30 -H "User-Agent: Mozilla/5.0" "{url}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=35)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f'curl failed for {series_id} (rc={result.returncode}): {result.stderr.strip()}'
        )
    df = pd.read_csv(StringIO(result.stdout))
    df.columns = ['date', 'value']
    df['date']  = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[(df['date'] >= START) & (df['date'] <= END)].set_index('date')['value']

    # Resample to monthly end-of-period (last non-NaN observation in each month).
    # This collapses daily series (e.g. oil) to a single row per month.
    monthly = df.resample('ME').last()
    monthly.index.name = 'date'
    return monthly


frames = {name: fetch(sid) for name, sid in SERIES.items()}
panel  = pd.DataFrame(frames)

panel['cpi_yoy'] = panel['cpi_level'].pct_change(12, fill_method=None) * 100
panel['pce_yoy'] = panel['pce_level'].pct_change(12, fill_method=None) * 100

os.makedirs('data/raw', exist_ok=True)
panel.to_csv('data/raw/fred_data.csv')
print(f"Saved data/raw/fred_data.csv — {panel.shape[0]} rows, {panel.shape[1]} cols")
print(panel.tail(3).to_string())
