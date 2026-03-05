"""
00_download_wrds_data.py
========================
Downloads all required data from WRDS for replication.
Requires: valid WRDS account, wrds Python package (pip install wrds)

Output:
    data/crsp_monthly.csv — CRSP monthly stock returns (2000-2024)
    data/ff_monthly.csv — Fama-French 5-factor monthly returns
    data/sic_mapping.csv — SIC codes from Audit Analytics
    data/company_tickers.csv — Company ticker mapping
"""
import wrds, pandas as pd, os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

db = wrds.Connection()

# ── CRSP Monthly Returns ──
print("Downloading CRSP monthly returns (2000-2024)...")
crsp = db.raw_sql("""
    SELECT a.permno, a.permco, a.date, a.ret, a.retx, a.vol, a.prc,
           a.shrout, a.cfacpr, a.cfacshr, b.ticker, b.comnam, b.shrcd, b.exchcd
    FROM crsp.msf AS a
    LEFT JOIN crsp.msenames AS b
        ON a.permno = b.permno
        AND a.date >= b.namedt AND a.date <= b.nameendt
    WHERE a.date BETWEEN '2000-01-01' AND '2024-12-31'
        AND b.shrcd IN (10, 11)
""")
crsp.to_csv(os.path.join(OUTPUT_DIR, 'crsp_monthly.csv'), index=False)
print(f"  CRSP: {len(crsp):,} rows, {crsp['permno'].nunique()} stocks")

# ── Fama-French 5-Factor Returns ──
print("Downloading FF5 factors...")
ff = db.raw_sql("""
    SELECT date, mktrf, smb, hml, rmw, cma, rf
    FROM ff.fivefactors_monthly
    WHERE date BETWEEN '2000-01-01' AND '2024-12-31'
""")
ff.to_csv(os.path.join(OUTPUT_DIR, 'ff_monthly.csv'), index=False)
print(f"  FF5: {len(ff)} months")

# ── SIC Codes from Audit Analytics ──
print("Downloading SIC codes from Audit Analytics...")
sic = db.raw_sql("""
    SELECT company_fkey, company_name AS name, sic
    FROM audit.auditnonreli
    WHERE sic IS NOT NULL
    GROUP BY company_fkey, company_name, sic
""")
sic.to_csv(os.path.join(OUTPUT_DIR, 'sic_mapping.csv'), index=False)
print(f"  SIC: {len(sic):,} firms")

# ── Company Tickers ──
print("Downloading company tickers...")
tickers = db.raw_sql("""
    SELECT DISTINCT company_fkey, best_edgar_ticker
    FROM audit.auditnonreli
    WHERE best_edgar_ticker IS NOT NULL
""")
tickers.to_csv(os.path.join(OUTPUT_DIR, 'company_tickers.csv'), index=False)
print(f"  Tickers: {len(tickers):,} firms")

db.close()
print("Done. All data saved to data/")
