import os
import re
import requests
import pandas as pd
import numpy as np

os.makedirs('data/raw', exist_ok=True)

# Try to load credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os as _os
KALSHI_EMAIL    = _os.getenv('KALSHI_EMAIL', '')
KALSHI_PASSWORD = _os.getenv('KALSHI_PASSWORD', '')

if not KALSHI_EMAIL or not KALSHI_PASSWORD or KALSHI_EMAIL == 'your_kalshi_login_email':
    print("Kalshi credentials not set — creating placeholder CSV.")
    print("To collect real data: set KALSHI_EMAIL and KALSHI_PASSWORD in .env and re-run this script.")
    dates = pd.date_range('2022-01-31', '2026-04-30', freq='ME')
    stub  = pd.DataFrame({'kalshi_implied_cpi': [np.nan] * len(dates)}, index=dates)
    stub.index.name = 'date'
    stub.to_csv('data/raw/kalshi_cpi.csv')
    print(f"Saved placeholder data/raw/kalshi_cpi.csv — {len(stub)} rows (all NaN)")
    exit(0)

BASE = 'https://trading-api.kalshi.com/trade-api/v2'

def login() -> dict:
    r = requests.post(f'{BASE}/login', json={'email': KALSHI_EMAIL, 'password': KALSHI_PASSWORD})
    r.raise_for_status()
    return {'Authorization': f'Bearer {r.json()["token"]}'}

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
    threshold_probs = []
    for m in markets_for_month:
        title = m.get('title', '') + ' ' + m.get('subtitle', '')
        match = re.search(r'(\d+\.?\d*)\s*%', title)
        if not match:
            continue
        threshold = float(match.group(1))
        result = m.get('result', '')
        prob = 1.0 if result == 'yes' else (0.0 if result == 'no' else float(m.get('yes_bid', 0.5)))
        threshold_probs.append((threshold, prob))
    if len(threshold_probs) < 2:
        return None
    threshold_probs.sort()
    thresholds = [tp[0] for tp in threshold_probs]
    probs      = [tp[1] for tp in threshold_probs]
    expected   = sum((probs[i] - probs[i+1]) * (thresholds[i] + thresholds[i+1]) / 2
                     for i in range(len(thresholds) - 1))
    expected  += (1.0 - probs[0]) * (thresholds[0] - 0.25)
    expected  += probs[-1] * (thresholds[-1] + 0.25)
    return round(expected, 3)

def main():
    headers     = login()
    all_series  = requests.get(f'{BASE}/series', headers=headers, params={'limit': 200}).json().get('series', [])
    cpi_series  = [s for s in all_series if 'CPI' in s.get('ticker','').upper()
                   or 'inflation' in s.get('title','').lower()]
    if not cpi_series:
        raise ValueError("No CPI series found on Kalshi.")
    series_ticker = cpi_series[0]['ticker']
    print(f"Using series: {series_ticker}")
    markets = get_all_markets(headers, series_ticker)
    print(f"Fetched {len(markets)} settled markets.")
    MONTH_RE = re.compile(
        r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+(\d{4})', re.IGNORECASE)
    by_month: dict = {}
    for m in markets:
        hit = MONTH_RE.search(m.get('title', ''))
        if hit:
            by_month.setdefault(hit.group(0), []).append(m)
    rows = []
    for month_str, mlist in sorted(by_month.items()):
        implied = implied_expected_cpi(mlist)
        if implied is not None:
            date = pd.to_datetime(month_str) + pd.offsets.MonthEnd(0)
            rows.append({'date': date, 'kalshi_implied_cpi': implied})
            print(f"  {month_str}: {implied}")
    df = pd.DataFrame(rows).set_index('date').sort_index()
    df = df[df.index >= '2022-01-31']
    df.to_csv('data/raw/kalshi_cpi.csv')
    print(f"Saved data/raw/kalshi_cpi.csv — {len(df)} rows")

if __name__ == '__main__':
    main()
