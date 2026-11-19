"""Generate manuscript tables and Figure 3 from stored analysis results."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
root,out=map(Path,sys.argv[1:3])
h=root/'results/holdout'
def primary(f):
    return pd.read_csv(f).query("scope=='primary' and k==3")
def ci(g):
    x=np.asarray(g);i=np.random.default_rng(42).integers(len(x),size=(2000,len(x)))
    return (x.mean(),*np.quantile(x[i].mean(axis=1),[.025,.975]))
flag='CHEMBL4190278'
orig=primary(h/'by_document.csv')
ach=orig[orig.target.eq('AChE')];gsk=orig[orig.target.eq('GSK3B')]
aug=primary(out/'augmented_by_document.csv').query("target=='AChE'")
panels={'AChE · strict':ach,'AChE · source-verified':aug,'AChE · document omitted':ach[ach.doc.ne(flag)],'GSK-3β · strict':gsk}
t2=[]; comps=[];contr=[]
for name,rows in panels.items():
    wide=rows.pivot(index='doc',columns='model',values='mae')
    for met in ['raw','mean']:
        gain,lo,hi=ci(wide[met]-wide.calibrated);contr.append([name,len(wide),met,gain,lo,hi])
    for model,g in rows.groupby('model'):
        m=g.mean(numeric_only=True);t2.append([name,len(g),model,m.mae,m.rho]);comps.append([name,model,m.mse,m.level_mse,m.within_mse])
pd.DataFrame(t2,columns=['panel','documents','model','mae','rho']).to_csv(out/'table2_by_panel.csv',index=False)
pd.DataFrame(contr,columns=['panel','documents','baseline','gain','low','high']).to_csv(out/'unit_panel_contrasts.csv',index=False)
cs=pd.DataFrame(comps,columns=['panel','model','mse','level','within']);cs.to_csv(out/'figure3_data.csv',index=False)
fig,axes=plt.subplots(2,2,sharex=True,sharey=True,figsize=(7.1,5),layout='constrained')
for ax,(name,df),letter in zip(axes.flat,cs.groupby('panel',sort=False),'ABCD'):
    v=df.set_index('model').loc[['raw','mean','calibrated']]
    ax.bar(range(3),v.level,.62,color='#245B70',label='Squared mean residual')
    ax.bar(range(3),v.within,.62,bottom=v.level,color='#B5D3D7',label='Residual variance')
    for i,(m,row) in enumerate(v.iterrows()):ax.text(i,row.mse+.03,f'{row.mse:.3f}',ha='center',fontsize=9)
    ax.set_title(letter+'   '+name,loc='left',fontsize=10,weight='bold')
    ax.set_xticks(range(3),['Raw','Ref. mean','Calibrated']);ax.tick_params(labelbottom=True)
    ax.set_ylim(0,2.02); ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    ax.spines[['right','top']].set_visible(False)
axes[0,1].legend(loc='upper right',fontsize=7,frameon=False)
fig.supylabel('Mean squared error (pIC₅₀²)',fontsize=10)
fig.savefig(root/'figures/Figure_3_Error_Decomposition.png',dpi=600,facecolor='white');plt.close(fig)
sens=pd.read_csv(out/'influence_and_provenance.csv')
t3=sens[sens.panel.isin(['four_flags_excluded','explicit_human_journal','unknown_and_patent_excluded','unknown_patent_and_outlier_excluded'])].copy()
t3.to_csv(out/'table3_both_contrasts.csv',index=False)
r=pd.read_csv(out/'reservation_released_by_document.csv')
for t,g in r.groupby('target'):
    w=g.pivot(index='doc',columns='model',values='mae')
    print('restoration',t,len(w),ci(w['mean']-w.calibrated),ci(w.raw-w.calibrated))
