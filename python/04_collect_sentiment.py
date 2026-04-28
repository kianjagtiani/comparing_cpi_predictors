import os
import time
import requests
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

def gdelt_monthly_tone(keyword: str, year: int, month: int) -> float:
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
    time.sleep(0.5)
    if not np.isnan(tone):
        print(f"  GDELT {d.strftime('%Y-%m')}: tone = {tone:.3f}")
    else:
        print(f"  GDELT {d.strftime('%Y-%m')}: no data")

gdelt_series = pd.DataFrame(gdelt_rows).set_index('date')['gdelt_tone']

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
