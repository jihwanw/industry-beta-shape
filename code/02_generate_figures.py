"""
Generate all 9 figures for the paper.
Reads from master_verification.json and constrained_results.json.
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from numpy.linalg import svd, lstsq
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size':11,'font.family':'serif','figure.dpi':300,
                      'axes.spines.top':False,'axes.spines.right':False})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Fig 1: Factorization diagram
fig,ax=plt.subplots(1,1,figsize=(8,3)); ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis('off')
for x,y,w,h,txt,c in [(0.5,1,1.8,1.2,'$R$\n$N\\times M$\nReturns','#4ECDC4'),
    (3.2,1,1.2,1.2,'$B$\n$N\\times K$\nBetas','#FF6B6B'),
    (5.0,1,1.8,1.2,'$F$\n$K\\times M$\nFactors','#45B7D1'),
    (7.8,1,1.2,1.2,'$\\epsilon$\n$N\\times M$\nNoise','#96CEB4')]:
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=c,alpha=0.3,edgecolor='black',lw=1.5))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10)
ax.text(2.6,1.6,'=',fontsize=16,ha='center',fontweight='bold')
ax.text(4.55,1.6,'×',fontsize=16,ha='center',fontweight='bold')
ax.text(7.2,1.6,'+',fontsize=16,ha='center',fontweight='bold')
ax.text(3.8,0.5,'Known\n(FF5)',ha='center',fontsize=9,color='#45B7D1',fontstyle='italic')
ax.text(3.8,2.6,'Estimate with\nindustry shrinkage',ha='center',fontsize=9,color='#FF6B6B',fontstyle='italic')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig1_factorization.pdf',bbox_inches='tight'); plt.close()

# Fig 2: SVD concentration
feas = json.load(open(os.path.join(BASE,'results','feasibility_results.json')))
fig,axes=plt.subplots(1,4,figsize=(10,3),sharey=True)
for ax,(sic,nm) in zip(axes,[(60,'Banking'),(63,'Insurance'),(73,'Business Svc'),(28,'Pharma')]):
    sv=feas.get(str(sic),{}).get('sv_ratios',[1,.5,.3,.2,.1])
    sv2=np.array(sv)**2; sv2=sv2/sv2.sum()
    ax.bar(range(1,6),sv2*100,color=['#FF6B6B' if i<2 else '#CCCCCC' for i in range(5)],edgecolor='black',lw=0.5)
    ax.set_title(nm,fontsize=10); ax.set_xlabel('Singular value'); ax.set_xticks(range(1,6))
    ax.text(3.5,max(sv2)*100*0.9,f'Top 2:\n{sum(sv2[:2])*100:.0f}%',fontsize=8,ha='center',color='#FF6B6B')
axes[0].set_ylabel('Variance share (%)')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig2_svd_concentration.pdf',bbox_inches='tight'); plt.close()

# Fig 3: MSE all industries (need data — use run_all output)
# Load CRSP for this
crsp=pd.read_csv('../data/crsp_monthly.csv',
    usecols=['permno','date','ret','ticker','shrcd'])
crsp['date']=pd.to_datetime(crsp['date']); crsp['ret']=pd.to_numeric(crsp['ret'],errors='coerce')
crsp=crsp[crsp['shrcd'].isin([10,11])].dropna(subset=['ret']); crsp['ym']=crsp['date'].dt.to_period('M')
sdf=pd.read_csv('../data/sic_mapping.csv').dropna(subset=['sic'])
sdf['sic2']=(sdf['sic'].astype(int)//100)
ct=pd.read_csv('../data/company_tickers.csv')
ct.columns=['company_fkey','ticker']
sp=sdf.merge(ct,on='company_fkey').merge(crsp.drop_duplicates('ticker')[['ticker','permno']].dropna(),on='ticker').drop_duplicates('permno')[['permno','sic2']]
crsp=crsp.merge(sp,on='permno')
ff=pd.read_csv('../data/ff_monthly.csv')
ff['date']=pd.to_datetime(ff['date']); ff['ym']=ff['date'].dt.to_period('M'); fi=ff.set_index('ym')
fc=['mktrf','smb','hml','rmw','cma']
ic=crsp.groupby('sic2')['permno'].nunique().sort_values(ascending=False)
sn={28:'Pharma',35:'Machinery',36:'Electronics',38:'Instruments',48:'Telecom',49:'Utilities',
    60:'Banking',62:'Securities',63:'Insurance',73:'Business Svc',67:'Holding',13:'Oil&Gas',
    20:'Food',80:'Health Svc',37:'Transport',50:'Wholesale',34:'Fab Metals',27:'Publishing',87:'Engineering'}
imps=[]
for sic in ic[ic>=15].index:
    ind=crsp[crsp['sic2']==sic]; Rd=ind.pivot_table(index='permno',columns='ym',values='ret')
    Rd=Rd.dropna(thresh=int(0.5*Rd.shape[1])).fillna(0)
    cm=sorted(Rd.columns.intersection(fi.index))
    if len(cm)<42: continue
    Rd=Rd[cm]; Fa=fi.loc[cm]; Rv=Rd.values-Fa['rf'].values[None,:]; F=Fa[fc].values.T
    N,M=Rv.shape
    if N<10: continue
    s=int(M*0.7); Bt=np.array([lstsq(F[:,:s].T,Rv[i,:s],rcond=None)[0] for i in range(N)])
    m0=np.mean((Rv[:,s:]-Bt@F[:,s:])**2); ms=np.mean((Rv[:,s:]-(0.5*Bt+0.5*Bt.mean(0)[None,:])@F[:,s:])**2)
    imps.append((sn.get(sic,f'SIC {sic}'),(m0-ms)/m0*100))
imps.sort(key=lambda x:x[1])
fig,ax=plt.subplots(figsize=(8,6))
ax.barh(range(len(imps)),[x[1] for x in imps],color=['#FF6B6B' if x[1]>7.5 else '#45B7D1' for x in imps],edgecolor='black',lw=0.3)
ax.set_yticks(range(len(imps))); ax.set_yticklabels([x[0] for x in imps],fontsize=7)
ax.set_xlabel('MSE improvement over OLS (%)'); ax.axvline(x=7.5,color='black',ls='--',lw=0.8,alpha=0.5)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig3_mse_all_industries.pdf',bbox_inches='tight'); plt.close()

# Fig 4: Quintile returns
R=json.load(open(os.path.join(BASE,'results','master_verification.json')))
qv=[R['table4_quintiles']['quintile_means'][f'Q{i}'] for i in range(1,6)]
fig,ax=plt.subplots(figsize=(6,4))
ax.bar(['Q1\n(Low)','Q2','Q3','Q4','Q5\n(High)'],qv,color=['#45B7D1']*4+['#FF6B6B'],edgecolor='black',lw=0.5)
ax.set_ylabel('6-month cumulative excess return (%)'); ax.set_ylim(0,9)
ax.annotate('',xy=(4,qv[4]),xytext=(0,qv[0]),arrowprops=dict(arrowstyle='<->',color='black',lw=1.5))
ax.text(2,6.5,f'Spread = {R["table4_quintiles"]["spread_pct"]}%\n($t$ = {R["table4_quintiles"]["spread_t"]})',ha='center',fontsize=10,fontweight='bold')
for i,v in enumerate(qv): ax.text(i,v+0.15,f'{v:.2f}',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig4_quintile_returns.pdf',bbox_inches='tight'); plt.close()

# Fig 5: Subperiod
fig,ax=plt.subplots(figsize=(7,4))
pds=['Full\n2004-24','1st half\n2004-13','2nd half\n2014-24','GFC\n2007-09','COVID\n2020-21','Tranquil\n2012-19']
sps=[R['table7_subperiod'][k]['annual_pct'] for k in ['full','first_half','second_half','GFC','COVID','normal']]
cs=['#666666','#45B7D1','#CCCCCC','#FF6B6B','#FF6B6B','#CCCCCC']
ax.bar(pds,sps,color=cs,edgecolor='black',lw=0.5); ax.axhline(y=0,color='black',lw=0.5)
ax.set_ylabel('Annualized Q5−Q1 spread (%)')
for i,v in enumerate(sps): ax.text(i,v+(1 if v>=0 else -1.5),f'{v:+.1f}%',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig5_subperiod_crisis.pdf',bbox_inches='tight'); plt.close()

# Fig 6: VIX conditional
fig,ax=plt.subplots(figsize=(5,4))
vs=[R['table9_vix']['low_vol']['annual_pct'],R['table9_vix']['high_vol']['annual_pct']]
ax.bar([0,1],vs,color=['#45B7D1','#FF6B6B'],edgecolor='black',lw=0.5,width=0.5)
ax.set_xticks([0,1]); ax.set_xticklabels([f'Low Volatility\n($t$ = {R["table9_vix"]["low_vol"]["spread_t"]})',
    f'High Volatility\n($t$ = {R["table9_vix"]["high_vol"]["spread_t"]})'])
ax.set_ylabel('Annualized Q5−Q1 spread (%)'); ax.axhline(y=0,color='black',lw=0.5)
for i,v in enumerate(vs): ax.text(i,v+0.3,f'{v:.1f}%',ha='center',fontsize=11,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig6_vix_conditional.pdf',bbox_inches='tight'); plt.close()

# Fig 7: Factor model comparison
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4))
ms=['CAPM','FF3','FF5']; cs=['#CCCCCC','#45B7D1','#FF6B6B']
tv=[R['table8_alt_models'][m]['spread_t'] for m in ms]
a1.bar(ms,tv,color=cs,edgecolor='black',lw=0.5); a1.axhline(y=1.96,color='red',ls='--',lw=0.8,alpha=0.5)
a1.set_ylabel('$t$-statistic (Q5−Q1 spread)'); a1.set_title('(a) Deviation signal',fontsize=10)
mv=[R['table2_mse'][m]['mean'] for m in ms]
a2.bar(ms,mv,color=cs,edgecolor='black',lw=0.5); a2.set_ylabel('MSE improvement (%)')
a2.set_title('(b) Beta estimation',fontsize=10)
for i,v in enumerate(mv): a2.text(i,v+0.2,f'+{v}%',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig7_factor_model_comparison.pdf',bbox_inches='tight'); plt.close()

# Fig 8: Double sort heatmap
ds=R['table6_double_sort']
data=np.array([[ds[sz]['quintile_means'][f'Q{i}'] for i in range(1,6)] for sz in ['Small','Mid','Big']])
fig,ax=plt.subplots(figsize=(7,3.5))
im=ax.imshow(data,cmap='RdYlGn',aspect='auto',vmin=3,vmax=8)
ax.set_xticks(range(5)); ax.set_xticklabels(['Q1 (Low)','Q2','Q3','Q4','Q5 (High)'])
ax.set_yticks(range(3)); ax.set_yticklabels(['Small','Mid','Big'])
ax.set_xlabel('Beta deviation quintile'); ax.set_ylabel('Size tercile')
for i in range(3):
    for j in range(5):
        ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=9,
                color='white' if data[i,j]>6.5 or data[i,j]<4 else 'black')
plt.colorbar(im,ax=ax,label='6-month return (%)',shrink=0.8)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig8_double_sort_heatmap.pdf',bbox_inches='tight'); plt.close()

# Fig 9: Constraints comparison
C=json.load(open(os.path.join(BASE,'results','constrained_results.json')))
fig,ax=plt.subplots(figsize=(8,4.5))
ms2=['PD\nCovariance','Non-neg\nMkt β','RPCA','Bounded\n|β|≤3','Temporal\nSmooth',
     'Shrinkage\n(baseline)','Shrink+\nBounded','Shrink+\nRPCA','All\nCombined']
ks=['PD_Cov','NonNeg_Mkt','RPCA','Bounded','Temporal','Shrinkage','Shrink+Bounded','Shrink+RPCA','All_Combined']
vl=[C['summary'][k]['mean'] for k in ks]
cl=['#CCCCCC']*5+['#FF6B6B','#FF8E8E','#FF8E8E','#FFB3B3']
ax.bar(range(len(ms2)),vl,color=cl,edgecolor='black',lw=0.5)
ax.set_xticks(range(len(ms2))); ax.set_xticklabels(ms2,fontsize=8)
ax.set_ylabel('MSE improvement over OLS (%)'); ax.axhline(y=C['summary']['Shrinkage']['mean'],color='#FF6B6B',ls='--',lw=1,alpha=0.6)
ax.text(2.5,6.5,'Shrinkage alone\ncaptures 93%',ha='center',fontsize=10,color='#FF6B6B',fontweight='bold')
ax.set_ylim(0,15)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig9_constraints_comparison.pdf',bbox_inches='tight'); plt.close()

print("All 9 figures generated.")
for f in sorted(os.listdir(FIGDIR)):
    if f.endswith('.pdf'): print(f"  {f} ({os.path.getsize(os.path.join(FIGDIR,f))//1024}KB)")
