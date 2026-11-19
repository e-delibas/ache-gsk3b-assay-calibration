"""Run with the archive's Python environment. Pass the extracted repository path."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau
from sklearn.model_selection import GroupKFold
arg = argparse.ArgumentParser()
arg.add_argument('repository', type=Path)
arg.add_argument('--output', type=Path, default=Path('analysis/outputs'))
args = arg.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
root = args.repository.resolve()
sys.path.insert(0,str(root/'src'))
from pilot import stable_seed, predict
def save(df,name): df.to_csv(args.output/(name+'.csv'), index=False)

def sample(group):
    rng = np.random.default_rng(stable_seed(group.assay.iloc[0]) % 2**32)
    for _ in range(20):
        order = rng.permutation(len(group))
        yield order[:3], order[3:]

def interval(values):
    x = np.asarray(values)
    draws = np.random.default_rng(42).integers(len(x), size=(2000,len(x)))
    lo, hi = np.quantile(x[draws].mean(axis=1), [.025,.975])
    return dict(n=len(x), gain=x.mean(), median=np.median(x), lower=lo,upper=hi,
                wins=(x>0).sum(), sign_p=binomtest((x>0).sum(), (x!=0).sum()).pvalue)
frame=pd.read_csv(root/'data/measurements.csv')
with np.load(root/'data/features.npz') as z:
    features=np.column_stack([z['fp'],z['desc']]).astype('float32')
    frame['fid']=pd.Index(z['smiles']).get_indexer(frame.smiles)
    fp=z['fp']
scores=[]; fitted=[]
for target,df in frame.groupby('target'):
    df=df.reset_index(drop=True)
    test=df.doc.map(lambda x:stable_seed(x)%5==0)
    x=features[df.fid]; y=df.pic50.to_numpy(); train=df.loc[~test]
    obs=df.loc[test].copy()
    obs['raw']=predict(np.flatnonzero(~test),np.flatnonzero(test),x,y)
    grid=np.linspace(0,1,21); curves=[]
    for tr,va in GroupKFold(3).split(train,groups=train.doc):
        train_center=train.iloc[tr].pic50.mean()
        for (doc,assay), rows in train.iloc[va].groupby(['doc','assay']):
            if len(rows)<10: continue
            vals=rows.pic50.to_numpy()
            losses=[]
            for r,q in sample(rows):
                forecast=train_center+grid*(vals[r].mean()-train_center)
                losses.append(np.abs(vals[q,None]-forecast).mean(axis=0))
            curves.append([doc,assay,*np.mean(losses,axis=0)])
    loss=pd.DataFrame(curves,columns=['doc','assay',*range(21)]).groupby('doc')[list(range(21))].mean().mean()
    lam=grid[loss.argmin()]
    fitted.append(dict(target=target,reference_weight=lam))
    w=pd.read_csv(root/'results/holdout/weights.csv').set_index('target').loc[target,'weight']
    for (doc,assay),rows in obs.groupby(['doc','assay']):
        if len(rows)<6: continue
        vals=rows.pic50.to_numpy();raw=rows.raw.to_numpy()
        bits=fp[rows.fid].astype(float); inter=bits@bits.T; counts=bits.sum(axis=1)
        sim=inter/np.maximum(1,counts[:,None]+counts[None,:]-inter)
        for rep,(r,q) in enumerate(sample(rows)):
            avg=vals[r].mean(); tc=train.pic50.mean()
            near=sim[np.ix_(q,r)]; mask=near==near.max(axis=1,keepdims=True)
            preds={'raw':raw[q],'mean':avg,'calibrated':avg+w*(raw[q]-raw[r].mean()),
                   'train_mean':tc,'ref_median':np.median(vals[r]),'shrinkage':tc+lam*(avg-tc),
                   'oracle_query_mean':vals[q].mean(),'oracle_query_median':np.median(vals[q]),
                   'nearest_reference':mask@vals[r]/mask.sum(axis=1)}
            for method,p in preds.items():
                err=p-vals[q]
                tau=kendalltau(vals[q],p).statistic if np.ptp(p)>0 and np.ptp(vals[q])>0 else np.nan
                scores.append([target,doc,assay,len(rows),rep,method,err.mean(),np.abs(err).mean(),np.mean(err**2),tau])
draw=pd.DataFrame(scores,columns=['target','doc','assay','size','repeat','method','bias','mae','mse','tau_b'])
for label,rows in [('primary',draw[draw['size']>=10]),('small',draw[draw['size']<10])]:
    docs=rows.groupby(['target','doc','assay','method'])[['bias','mae','mse']].mean().groupby(['target','doc','method']).mean().reset_index()
    save(docs,label+'_by_document')
    summary=docs.groupby(['target','method']).mean(numeric_only=True).reset_index();save(summary,label+'_summary')
    print(label,docs.groupby('target').doc.nunique().to_dict(),summary.pivot(index='target',columns='method',values='mae').round(4).to_string())
    if label=='primary':
        original=pd.read_csv(root/'results/holdout/by_document.csv').query("scope=='primary' and k==3")
        chk=original.merge(docs,left_on=['target','doc','model'],right_on=['target','doc','method'])
        np.testing.assert_allclose(chk.mae_x,chk.mae_y,atol=1e-9)
        print(docs[docs.doc.eq('CHEMBL4190278')].round(4).to_string(index=False))
        paired=rows.pivot(index=['target','doc','assay','repeat'],columns='method',values='tau_b')[['raw','nearest_reference']].dropna()
        dif=(paired.nearest_reference-paired.raw).groupby(['target','doc','assay']).mean().groupby(['target','doc']).mean()
        print('tau-b',[(k,interval(g)) for k,g in dif.groupby(level='target')])
        save(pd.DataFrame([dict(target=k,**interval(g)) for k,g in dif.groupby(level='target')]),'tau_b_matched')
save(pd.DataFrame(fitted),'shrinkage_weights')
print(fitted)
