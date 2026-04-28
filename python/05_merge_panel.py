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
panel = panel.drop(columns=['cpi_level', 'pce_level'], errors='ignore')

panel['cpi_yoy_next'] = panel['cpi_yoy'].shift(-1)

panel = panel.loc['2022-01-31':'2026-03-31']
panel = panel.dropna(subset=['cpi_yoy_next'])

os.makedirs('data/processed', exist_ok=True)
panel.to_csv('data/processed/monthly_panel.csv')

print(f"Saved data/processed/monthly_panel.csv")
print(f"Shape: {panel.shape[0]} rows x {panel.shape[1]} cols")
print(f"\nColumns: {list(panel.columns)}")
print(f"\nMissing values:\n{panel.isnull().sum().to_string()}")
print(f"\nDate range: {panel.index[0].date()} to {panel.index[-1].date()}")
