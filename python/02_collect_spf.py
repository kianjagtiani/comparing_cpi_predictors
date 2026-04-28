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

    xl = xl[['YEAR', 'QUARTER', 'CPI2']].copy()
    xl.columns = ['year', 'quarter', 'spf_cpi_forecast']
    xl = xl.dropna(subset=['spf_cpi_forecast'])

    def to_month_end(row):
        month = int(row['quarter']) * 3
        return pd.Timestamp(year=int(row['year']), month=month, day=1) + pd.offsets.MonthEnd(0)

    xl['date'] = xl.apply(to_month_end, axis=1)
    xl = xl[xl['date'] >= '2021-01-01'].set_index('date')[['spf_cpi_forecast']]

    monthly_idx = pd.date_range('2021-01-31', '2026-04-30', freq='ME')
    xl = xl.reindex(monthly_idx).ffill()
    xl.index.name = 'date'
    return xl

df = load_spf()
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/spf_data.csv')
print(f"Saved data/raw/spf_data.csv — {df.shape[0]} rows")
print(df.tail(6).to_string())
