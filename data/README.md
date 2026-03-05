# Data Directory

This directory stores raw data downloaded from WRDS. Files are not included in the repository due to licensing restrictions.

## How to obtain the data

Run the download script with a valid WRDS account:

```bash
python code/00_download_wrds_data.py
```

## Expected files after download

| File | Description | Source |
|------|-------------|--------|
| `crsp_monthly.csv` | CRSP monthly stock returns (2000–2024) | CRSP via WRDS |
| `ff_monthly.csv` | Fama-French 5-factor monthly returns | FF via WRDS |
| `sic_mapping.csv` | SIC codes by company | Audit Analytics via WRDS |
| `company_tickers.csv` | Company-ticker mapping | Audit Analytics via WRDS |
