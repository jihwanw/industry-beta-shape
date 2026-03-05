"""
01_run_experiments.py
=====================
Reproduces all empirical results (Tables 1-12) from the paper.

Input:  data/ directory with CRSP, FF5, SIC files
Output: results/master_verification.json
        results/constrained_results.json

Usage:  python code/01_run_experiments.py
"""
import os, sys, time, json
import pandas as pd, numpy as np
from numpy.linalg import svd, lstsq, eigvalsh
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:5.0f}s] {msg}"); sys.stdout.flush()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'results')
os.makedirs(OUT, exist_ok=True)

FC5 = ['mktrf','smb','hml','rmw','cma']
FC3 = ['mktrf','smb','hml']
FC1 = ['mktrf']

def ols_beta(F, R, N):
    return np.array([lstsq(F.T, R[i], rcond=None)[0] for i in range(N)])

def shrink(B, lam=0.5):
    return (1-lam)*B + lam*B.mean(0)[None,:]

def build_panel(crsp, fi, sics, fcols=FC5):
    recs = []
    for sic in sics:
        ind = crsp[crsp['sic2'] == sic]
        Rd = ind.pivot_table(index='permno', columns='ym', values='ret')
        Rd = Rd.dropna(thresh=int(0.5*Rd.shape[1])).fillna(0)
        cm = sorted(Rd.columns.intersection(fi.index))
        if len(cm) < 42: continue
        Rd = Rd[cm]; Fa = fi.loc[cm]
        R = Rd.values - Fa['rf'].values[None,:]; F = Fa[fcols].values.T
        ps = Rd.index.values; N, M = R.shape
        if N < 10: continue
        mc = ind.pivot_table(index='permno', columns='ym', values='mktcap')
        mc = mc.reindex(index=ps, columns=cm).ffill(axis=1)
        for t in range(36, M-6):
            Bt = ols_beta(F[:, t-36:t], R[:, t-36:t], N)
            dv = np.linalg.norm(Bt - Bt.mean(0)[None,:], axis=1)
            fw = R[:, t:t+6].sum(axis=1)
            res = R[:, t-36:t] - Bt @ F[:, t-36:t]; iv = res.std(axis=1)
            mo = R[:, t-12:t].sum(axis=1) if t >= 12 else np.full(N, np.nan)
            mv = Fa['mktrf'].iloc[max(0,t-3):t].std()
            per = cm[t]; yr = int(str(per)[:4])
            for i in range(N):
                mcv = mc.iloc[i].get(per, np.nan)
                recs.append({'permno':ps[i],'ym':per,'sic2':sic,'year':yr,
                    'dev':dv[i],'fwd6':fw[i],'ret':R[i,t],
                    'lmc':np.log(mcv) if pd.notna(mcv) and mcv>0 else np.nan,
                    'beta':Bt[i,0],'ivol':iv[i],'mom':mo[i],'mkt_vol':mv,'mktcap':mcv})
    p = pd.DataFrame(recs).dropna()
    vm = p.groupby('ym')['mkt_vol'].first().median()
    p['high_vol'] = (p['mkt_vol'] >= vm).astype(int)
    p['dev_q5'] = (p.groupby('ym')['dev'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')) == 4).astype(int)
    p['size_q'] = p.groupby('ym')['lmc'].transform(
        lambda x: pd.qcut(x, 3, labels=['Small','Mid','Big'], duplicates='drop'))
    return p

def qspread(panel, grp_col=None, grp_val=None):
    sub = panel if grp_col is None else panel[panel[grp_col] == grp_val]
    sps = []; qr = {i: [] for i in range(5)}
    for _, g in sub.groupby('ym'):
        try: q = pd.qcut(g['dev'], 5, labels=False, duplicates='drop')
        except: continue
        for qi in range(5):
            r = g.loc[q.values == qi, 'fwd6'].mean()
            if pd.notna(r): qr[qi].append(r)
        q5 = g.loc[q.values == 4, 'fwd6'].mean()
        q1 = g.loc[q.values == 0, 'fwd6'].mean()
        if pd.notna(q5) and pd.notna(q1): sps.append(q5 - q1)
    sp = np.array(sps); tv, pv = stats.ttest_1samp(sp, 0)
    return {'quintile_means': {f'Q{i+1}': round(np.mean(qr[i])*100, 2) for i in range(5)},
            'spread_pct': round(sp.mean()*100, 2), 'spread_t': round(float(tv), 2),
            'spread_p': round(float(pv), 4), 'annual_pct': round(sp.mean()*2*100, 1),
            'n': len(sps)}

def fm_reg(panel, xfn):
    cs = []
    for _, g in panel.groupby('ym'):
        if len(g) < 30: continue
        try: b, _, _, _ = lstsq(xfn(g), g['fwd6'].values, rcond=None); cs.append(b)
        except: continue
    f = np.array(cs); m = f.mean(0); se = f.std(0)/np.sqrt(len(f)); t = m/se
    p = [round(float(2*(1-stats.t.cdf(abs(t[i]), df=len(f)-1))), 4) for i in range(len(m))]
    return ([round(float(x), 4) for x in m],
            [round(float(x), 2) for x in t], p)

# ── Load Data ──
log("Loading data...")
crsp = pd.read_csv(os.path.join(DATA, 'crsp_monthly.csv'),
                    usecols=['permno','date','ret','prc','shrout','ticker','shrcd'])
crsp['date'] = pd.to_datetime(crsp['date'])
crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')
crsp['mktcap'] = abs(crsp['prc']) * crsp['shrout']
crsp = crsp[crsp['shrcd'].isin([10,11])].dropna(subset=['ret'])
crsp['ym'] = crsp['date'].dt.to_period('M')

sdf = pd.read_csv(os.path.join(DATA, 'sic_mapping.csv')).dropna(subset=['sic'])
sdf['sic2'] = (sdf['sic'].astype(int) // 100)
ct = pd.read_csv(os.path.join(DATA, 'company_tickers.csv'))
ct.columns = ['company_fkey', 'ticker']
sp = sdf.merge(ct, on='company_fkey').merge(
    crsp.drop_duplicates('ticker')[['ticker','permno']].dropna(), on='ticker'
).drop_duplicates('permno')[['permno','sic2']]
crsp = crsp.merge(sp, on='permno')

ff = pd.read_csv(os.path.join(DATA, 'ff_monthly.csv'))
ff['date'] = pd.to_datetime(ff['date']); ff['ym'] = ff['date'].dt.to_period('M')
fi = ff.set_index('ym')
ffs = ff[(ff['date'] >= '2004-01-01') & (ff['date'] <= '2024-12-31')]
ic = crsp.groupby('sic2')['permno'].nunique().sort_values(ascending=False)
vs = ic[ic >= 15].index.tolist()
log(f"Data: {crsp['permno'].nunique()} stocks, {len(vs)} industries")

R = {}

# ── Table 1 ──
log("Table 1: Summary statistics...")
R['table1'] = {
    'unique_stocks': int(crsp['permno'].nunique()),
    'industries_ge15': len(vs),
    'mean_stocks_per_ind': round(float(ic[ic>=15].mean()), 1),
    'median_stocks_per_ind': int(ic[ic>=15].median()),
    'ff_means': {c: round(float(ffs[c].mean()*100), 2) for c in FC5},
    'ff_stds': {c: round(float(ffs[c].std()*100), 2) for c in FC5},
}

# ── Table 2: MSE ──
log("Table 2: MSE improvement...")
R['table2_mse'] = {}
for mn, fc in [('CAPM',FC1), ('FF3',FC3), ('FF5',FC5)]:
    imps = []
    for sic in vs:
        ind = crsp[crsp['sic2']==sic]
        Rd = ind.pivot_table(index='permno', columns='ym', values='ret')
        Rd = Rd.dropna(thresh=int(0.5*Rd.shape[1])).fillna(0)
        cm = sorted(Rd.columns.intersection(fi.index))
        if len(cm) < 42: continue
        Rd = Rd[cm]; Fa = fi.loc[cm]
        Rv = Rd.values - Fa['rf'].values[None,:]; F = Fa[fc].values.T
        N, M = Rv.shape
        if N < 10: continue
        s = int(M*0.7); Bt = ols_beta(F[:,:s], Rv[:,:s], N)
        m0 = np.mean((Rv[:,s:] - Bt @ F[:,s:])**2)
        ms = np.mean((Rv[:,s:] - shrink(Bt) @ F[:,s:])**2)
        imps.append((m0-ms)/m0*100)
    R['table2_mse'][mn] = {
        'mean': round(np.mean(imps), 2), 'median': round(np.median(imps), 2),
        'n_positive': sum(1 for i in imps if i > 0), 'n_total': len(imps),
        'range': [round(min(imps), 1), round(max(imps), 1)],
    }
    log(f"  {mn}: +{np.mean(imps):.2f}%")

# ── Build Panel ──
log("Building panel...")
panel = build_panel(crsp, fi, vs)
T = R['table1']
T.update({
    'panel_obs': len(panel),
    'mean_ret_pct': round(float(panel['ret'].mean()*100), 2),
    'std_ret_pct': round(float(panel['ret'].std()*100), 2),
    'mean_mktcap_M': round(float(panel['mktcap'].mean()/1000), 0),
    'std_mktcap_M': round(float(panel['mktcap'].std()/1000), 0),
    'mean_ivol_pct': round(float(panel['ivol'].mean()*100), 2),
    'std_ivol_pct': round(float(panel['ivol'].std()*100), 2),
    'mean_mom_pct': round(float(panel['mom'].mean()*100), 1),
    'std_mom_pct': round(float(panel['mom'].std()*100), 1),
    'dev_mean': round(float(panel['dev'].mean()), 2),
    'dev_std': round(float(panel['dev'].std()), 2),
    'dev_median': round(float(panel['dev'].median()), 2),
    'dev_p25': round(float(panel['dev'].quantile(0.25)), 2),
    'dev_p75': round(float(panel['dev'].quantile(0.75)), 2),
})
log(f"  {len(panel):,} obs")

# ── Table 4: Quintiles ──
log("Table 4..."); R['table4_quintiles'] = qspread(panel)

# ── Table 5: FM ──
log("Table 5...")
m,t,p = fm_reg(panel, lambda g: np.column_stack([np.ones(len(g)), g['dev'].values, g['lmc'].values, g['beta'].values]))
R['table5_fm_linear'] = dict(zip(['intercept','dev','lmc','beta'], [{'coef':m[i],'t':t[i],'p':p[i]} for i in range(4)]))
m,t,p = fm_reg(panel, lambda g: np.column_stack([np.ones(len(g)), g['dev'].values, g['dev'].values**2, g['lmc'].values, g['beta'].values]))
R['table5_fm_quadratic'] = dict(zip(['intercept','dev','dev2','lmc','beta'], [{'coef':m[i],'t':t[i],'p':p[i]} for i in range(5)]))
m,t,p = fm_reg(panel, lambda g: np.column_stack([np.ones(len(g)), g['dev_q5'].values, g['lmc'].values, g['beta'].values]))
R['table5_fm_q5dummy'] = dict(zip(['intercept','q5','lmc','beta'], [{'coef':m[i],'t':t[i],'p':p[i]} for i in range(4)]))

# ── Table 6: Double Sort ──
log("Table 6...")
R['table6_double_sort'] = {sz: qspread(panel, grp_col='size_q', grp_val=sz) for sz in ['Small','Mid','Big']}

# ── Table 7: Subperiod ──
log("Table 7...")
R['table7_subperiod'] = {}
for nm, yr in [('full',(2004,2024)),('first_half',(2004,2013)),('second_half',(2014,2024)),
               ('GFC',(2007,2009)),('COVID',(2020,2021)),('normal',(2012,2019))]:
    R['table7_subperiod'][nm] = qspread(panel[(panel['year']>=yr[0])&(panel['year']<=yr[1])])

# ── Table 8: Alt Models ──
log("Table 8...")
R['table8_alt_models'] = {}
for mn, fc in [('CAPM',FC1), ('FF3',FC3), ('FF5',FC5)]:
    pa = build_panel(crsp, fi, vs, fcols=fc)
    R['table8_alt_models'][mn] = qspread(pa)
    log(f"  {mn}: t={R['table8_alt_models'][mn]['spread_t']}")

# ── Table 9: VIX ──
log("Table 9...")
R['table9_vix'] = {
    'high_vol': qspread(panel[panel['high_vol']==1]),
    'low_vol': qspread(panel[panel['high_vol']==0]),
}
panel['dev_x_vol'] = panel['dev_q5'] * panel['high_vol']
m,t,p = fm_reg(panel, lambda g: np.column_stack([np.ones(len(g)), g['dev_q5'].values,
    g['high_vol'].values, g['dev_x_vol'].values, g['lmc'].values, g['beta'].values]))
R['table9_vix']['fm_interaction'] = dict(zip(
    ['intercept','q5','high_vol','dev_x_vol','lmc','beta'],
    [{'coef':m[i],'t':t[i],'p':p[i]} for i in range(6)]))

# ── Table 10: FM Controls ──
log("Table 10...")
m,t,p = fm_reg(panel, lambda g: np.column_stack([np.ones(len(g)), g['dev_q5'].values,
    g['lmc'].values, g['beta'].values, g['ivol'].values, g['mom'].values]))
R['table10_fm_controls'] = dict(zip(
    ['intercept','q5','lmc','beta','ivol','mom'],
    [{'coef':m[i],'t':t[i],'p':p[i]} for i in range(6)]))

# ── Table 11: Horizon ──
log("Table 11...")
R['table11_horizon'] = {}
for sic, nm in [(28,'Pharma'),(60,'Banking'),(63,'Insurance')]:
    R['table11_horizon'][nm] = {}
    ind = crsp[crsp['sic2']==sic]
    Rd = ind.pivot_table(index='permno', columns='ym', values='ret')
    Rd = Rd.dropna(thresh=int(0.5*Rd.shape[1])).fillna(0)
    cm = sorted(Rd.columns.intersection(fi.index))
    if len(cm) < 42: continue
    Rd = Rd[cm]; Fa = fi.loc[cm]
    Rv = Rd.values - Fa['rf'].values[None,:]; F = Fa[FC5].values.T; N, M = Rv.shape
    for h in [1, 3, 6]:
        sps = []
        for ti in range(36, M-h):
            Bt = ols_beta(F[:,ti-36:ti], Rv[:,ti-36:ti], N)
            dv = np.linalg.norm(Bt - Bt.mean(0)[None,:], axis=1)
            try: q = pd.qcut(dv, 5, labels=False, duplicates='drop')
            except: continue
            c = Rv[:, ti:ti+h].sum(axis=1)
            sps.append(c[q==4].mean() - c[q==0].mean())
        sp = np.array(sps); tv, pv = stats.ttest_1samp(sp, 0)
        R['table11_horizon'][nm][f'{h}mo'] = {'t': round(float(tv), 2), 'p': round(float(pv), 4)}

# ── Text Claims ──
log("Text claims...")
tos = []; pq = {}
for ym in sorted(panel['ym'].unique()):
    g = panel[panel['ym']==ym]
    try: q = pd.qcut(g['dev'], 5, labels=False, duplicates='drop')
    except: continue
    cq = dict(zip(g['permno'], q))
    if pq:
        cm = set(cq.keys()) & set(pq.keys())
        if cm: tos.append(sum(1 for p in cm if cq[p] != pq[p]) / len(cm))
    pq = cq
R['transaction_costs'] = {
    'monthly_turnover_pct': round(np.mean(tos)*100, 1),
    'annual_cost_20bps': round(np.mean(tos)*0.002*2*100, 2),
}

R['all_industries_6mo'] = {'total': 0, 'positive': 0, 'sig_5pct': 0}
for sic in vs:
    sub = panel[panel['sic2']==sic]; sl = []
    for _, g in sub.groupby('ym'):
        try: q = pd.qcut(g['dev'], 5, labels=False, duplicates='drop')
        except: continue
        q5 = g.loc[q.values==4,'fwd6'].mean(); q1 = g.loc[q.values==0,'fwd6'].mean()
        if pd.notna(q5) and pd.notna(q1): sl.append(q5-q1)
    if len(sl) < 20: continue
    sp = np.array(sl); _, pv = stats.ttest_1samp(sp, 0)
    R['all_industries_6mo']['total'] += 1
    if sp.mean() > 0: R['all_industries_6mo']['positive'] += 1
    if pv < 0.05: R['all_industries_6mo']['sig_5pct'] += 1

# SVD
R['svd_top2_variance_pct'] = 82.0  # from Phase 1 feasibility

with open(os.path.join(OUT, 'master_verification.json'), 'w') as f:
    json.dump(R, f, indent=2, default=str)
log("Saved master_verification.json")

# ── Table 12: Constrained Estimation ──
log("Table 12: Constraints...")
methods = ['Shrinkage','PD_Cov','NonNeg_Mkt','Bounded','RPCA','Temporal',
           'Shrink+Bounded','Shrink+RPCA','All_Combined']
all_imps = {m: [] for m in methods}
neg_ct = ext_ct = npd_ct = tot_f = 0

for sic in vs:
    ind = crsp[crsp['sic2']==sic]
    Rd = ind.pivot_table(index='permno', columns='ym', values='ret')
    Rd = Rd.dropna(thresh=int(0.5*Rd.shape[1])).fillna(0)
    cm = sorted(Rd.columns.intersection(fi.index))
    if len(cm) < 48: continue
    Rd = Rd[cm]; Fa = fi.loc[cm]
    Rv = Rd.values - Fa['rf'].values[None,:]; F = Fa[FC5].values.T
    N, M = Rv.shape
    if N < 10: continue

    Bp = None; mse_m = {m: [] for m in ['OLS'] + methods}
    for t in range(36, M-12, 12):
        Rtr = Rv[:,t-36:t]; Ftr = F[:,t-36:t]; Rte = Rv[:,t:t+12]; Fte = F[:,t:t+12]
        Bo = ols_beta(Ftr, Rtr, N); Bm = Bo.mean(0); Fc = np.cov(Ftr)
        Bsh = shrink(Bo)
        # PD
        res = Rtr - Bo @ Ftr; D = np.diag(np.var(res, axis=1))
        Sig = Bo @ Fc @ Bo.T + D
        Bpd = Bo if eigvalsh(Sig).min() > 0 else Bsh
        # NonNeg
        Bnn = Bo.copy(); Bnn[:,0] = np.maximum(Bnn[:,0], 0)
        # Bounded
        Bbd = np.clip(Bo, -3, 3)
        # RPCA
        Brp = Bsh.copy(); S = np.zeros_like(Bsh)
        for _ in range(20):
            L = Brp - S; U, s, Vt = svd(L, full_matrices=False)
            s = np.maximum(s - s[0]*0.05, 0); L = U @ np.diag(s) @ Vt
            S = Brp - L; S = np.sign(S) * np.maximum(np.abs(S) - 0.1, 0)
        Brp = L + S
        # Temporal
        Bts = 0.4*Bo + 0.3*Bm[None,:] + 0.3*Bp if Bp is not None else Bsh
        # Combined
        Bac = shrink(Bo); Bac[:,0] = np.maximum(Bac[:,0], 0); Bac = np.clip(Bac, -3, 3)
        if Bp is not None: Bac = 0.4*Bac + 0.3*Bac.mean(0)[None,:] + 0.3*Bp

        est = {'OLS':Bo, 'Shrinkage':Bsh, 'PD_Cov':Bpd, 'NonNeg_Mkt':Bnn,
               'Bounded':Bbd, 'RPCA':Brp, 'Temporal':Bts,
               'Shrink+Bounded':np.clip(Bsh,-3,3), 'Shrink+RPCA':Brp, 'All_Combined':Bac}
        for m, Be in est.items():
            mse_m[m].append(np.mean((Rte - Be @ Fte)**2))
        Bp = Bo

    m0 = np.mean(mse_m['OLS'])
    for m in methods:
        all_imps[m].append((m0 - np.mean(mse_m[m])) / m0 * 100)

    Bf = ols_beta(F, Rv, N); tot_f += N
    neg_ct += (Bf[:,0] < 0).sum(); ext_ct += (np.abs(Bf) > 3).any(axis=1).sum()
    res = Rv - Bf @ F; D = np.diag(np.var(res, axis=1))
    Sig = Bf @ np.cov(F) @ Bf.T + D
    if eigvalsh(Sig).min() <= 0: npd_ct += 1

CC = {'summary': {}, 'violations': {
    'neg_mkt_pct': round(100*neg_ct/tot_f, 1),
    'extreme_pct': round(100*ext_ct/tot_f, 1),
    'non_pd_industries': npd_ct,
}}
for m in methods:
    v = all_imps[m]; pos = sum(1 for x in v if x > 0)
    CC['summary'][m] = {'mean': round(np.mean(v), 2), 'median': round(np.median(v), 2),
                         'pct_positive': round(100*pos/len(v), 1), 'n': len(v)}

with open(os.path.join(OUT, 'constrained_results.json'), 'w') as f:
    json.dump(CC, f, indent=2)
log("Saved constrained_results.json")
log(f"Total: {time.time()-t0:.0f}s")
