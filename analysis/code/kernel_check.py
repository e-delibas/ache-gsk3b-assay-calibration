"""Development-selected fingerprint kernel ridge benchmark for the review diagnostics."""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

arg=argparse.ArgumentParser()
arg.add_argument('repository',type=Path)
arg.add_argument('--output',type=Path,default=Path('analysis/outputs'))
args=arg.parse_args()
root=args.repository.resolve()
args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'src'))
from pilot import stable_seed

def predict(k, y, train, test, ridge):
    block=k[np.ix_(train,train)].copy()
    block.flat[::len(train)+1]+=ridge
    center=y[train].mean()
    return center+k[np.ix_(test,train)]@solve(block,y[train]-center,assume_a='pos')

def evaluate(rows, predictions, weights, with_rank=False):
    scored=[]
    for (doc,assay),g in rows.assign(pred=predictions).groupby(['doc','assay']):
        if len(g)<10:continue
        rng=np.random.default_rng(stable_seed(assay)%2**32)
        y=g.pic50.to_numpy();f=g.pred.to_numpy()
        losses=[];ranks=[]
        for _ in range(20):
            perm=rng.permutation(len(y));ref=perm[:3];q=perm[3:]
            mu=y[ref].mean(); out=mu+weights[:,None]*(f[q]-f[ref].mean())
            losses.append(np.abs(out-y[q]).mean(axis=1))
            if with_rank:ranks.append(spearmanr(f[q],y[q]).statistic)
        scored.append([doc,assay,*np.mean(losses,axis=0),np.mean(ranks) if ranks else np.nan])
    return pd.DataFrame(scored,columns=['doc','assay',*weights,'raw_rho']).groupby('doc').mean(numeric_only=True)

m=pd.read_csv(root/'data/measurements.csv');params=[];results=[]
with np.load(root/'data/features.npz') as z:
    m['fid']=pd.Index(z['smiles']).get_indexer(m.smiles); fp=z['fp']
for target,g in m.groupby('target'):
    g=g.reset_index(drop=True); bits=fp[g.fid].astype('float32'); inter=bits@bits.T; n=bits.sum(axis=1)
    k=(inter/np.maximum(n[:,None]+n[None,:]-inter,1)).astype('float64');y=g.pic50.to_numpy()
    mask=g.doc.map(lambda s:stable_seed(s)%5==0)
    train=np.flatnonzero(~mask);test=np.flatnonzero(mask); weights=np.arange(21)/20
    best=(float('inf'),None,None)
    for ridge in [.1,1.,10.]:
        oof=np.full(len(train),np.nan)
        for tr,va in GroupKFold(3).split(train,groups=g.iloc[train].doc):
            oof[va]=predict(k,y,train[tr],train[va],ridge)
        losses=evaluate(g.iloc[train],oof,weights)[list(weights)].mean()
        w=losses.idxmin()
        best=min(best,(losses.loc[w],ridge,w))
    loss,ridge,w=best
    tested=evaluate(g.iloc[test],predict(k,y,train,test,ridge),np.unique([0.,w]),True)
    gain=tested[0.]-tested[w]; idx=np.random.default_rng(42).integers(len(gain),size=(2000,len(gain)))
    lo,hi=np.quantile(gain.to_numpy()[idx].mean(axis=1),[.025,.975])
    vals=dict(target=target,development_loss=loss,ridge=ridge,weight=w,calibrated_mae=tested[w].mean(),mean_mae=tested[0.].mean(),raw_rho=tested.raw_rho.mean(),gain=gain.mean(),low=lo,high=hi)
    params.append(vals);print(vals,flush=True)
    results.append(tested.assign(target=target,reference_mae=tested[0.],calibrated_mae=tested[w],gain=gain)[['target','reference_mae','calibrated_mae','gain','raw_rho']].reset_index())
pd.DataFrame(params).to_csv(args.output/'kernel_summary.csv',index=False)
pd.concat(results).to_csv(args.output/'kernel_by_document.csv',index=False)
