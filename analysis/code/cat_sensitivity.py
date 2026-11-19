"""Small post hoc CatBoost grid, repeated seeds, and count-specific calibration weights."""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
arg=argparse.ArgumentParser()
arg.add_argument('repository',type=Path);arg.add_argument('--output',type=Path,default=Path('analysis/outputs'))
args=arg.parse_args(); root=args.repository.resolve(); args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'src'))
from pilot import stable_seed
weights=np.arange(21)/20

def model_pred(x,y,tr,te,depth,trees,seed):
    model=CatBoostRegressor(iterations=trees,depth=depth,learning_rate=.05,thread_count=4,random_seed=seed,
                            allow_writing_files=False,verbose=False)
    model.fit(x[tr],y[tr]);return model.predict(x[te])

def errors(rows,pred,k=3,common=False):
    records=[]
    for (doc,assay),g in rows.assign(pred=pred).groupby(['doc','assay']):
        if len(g)<10: continue
        rng=np.random.default_rng(stable_seed(assay)%2**32)
        yy=g.pic50.to_numpy();ff=g.pred.to_numpy()
        loss=[]
        for _ in range(20):
            p=rng.permutation(len(g));r=p[:k];q=p[5 if common else k:]
            yhat=yy[r].mean()+weights[:,None]*(ff[q]-ff[r].mean())
            loss.append(np.abs(yhat-yy[q]).mean(axis=1))
        records.append([doc,*np.mean(loss,axis=0)])
    return pd.DataFrame(records,columns=['doc',*weights]).groupby('doc').mean()
def result(target,design,seed,d,trees,k,weight,scored):
    gain=scored[0.]-scored[weight]; ids=np.random.default_rng(42).integers(len(gain),size=(2000,len(gain)))
    lo,hi=np.quantile(gain.to_numpy()[ids].mean(axis=1),[.025,.975])
    return dict(target=target,design=design,seed=seed,depth=d,trees=trees,k=k,w=weight,cal_mae=scored[weight].mean(),gain=gain.mean(),lo=lo,hi=hi)
m=pd.read_csv(root/'data/measurements.csv')
with np.load(root/'data/features.npz') as z:
    m['fid']=pd.Index(z['smiles']).get_indexer(m.smiles);feat=np.column_stack([z['fp'],z['desc']]).astype('float32')
out=[]
for target,g in m.groupby('target'):
    g=g.reset_index(drop=True);mask=g.doc.map(lambda s:stable_seed(s)%5==0);tr=np.flatnonzero(~mask);te=np.flatnonzero(mask)
    x=feat[g.fid];y=g.pic50.to_numpy()
    def oof(d,trees,seed):
        arr=np.zeros(len(tr))
        for train,val in GroupKFold(3).split(tr,groups=g.iloc[tr].doc):
            arr[val]=model_pred(x,y,tr[train],tr[val],d,trees,seed)
        return arr
    fits={}
    for d,trees in [(4,220),(6,220),(6,400)]:
        pred=oof(d,trees,42);loss=errors(g.iloc[tr],pred).mean()
        fits[(d,trees)]=(loss.min(),loss.idxmin(),pred)
    d,trees=min(fits,key=lambda key:fits[key][0])
    for seed in [42,17,73]:
        pred=fits[(d,trees)][2] if seed==42 else oof(d,trees,seed)
        weight=errors(g.iloc[tr],pred).mean().idxmin()
        testpred=model_pred(x,y,tr,te,d,trees,seed)
        out.append(result(target,'tuned_cat_primary',seed,d,trees,3,weight,errors(g.iloc[te],testpred)))
    original=fits[(6,220)][2]; testpred=model_pred(x,y,tr,te,6,220,42)
    for k in [1,3,5]:
        w=errors(g.iloc[tr],original,k,True).mean().idxmin()
        out.append(result(target,'original_cat_common_queries_retuned',42,6,220,k,w,errors(g.iloc[te],testpred,k,True)))
    print(target,out[-6:],flush=True)
pd.DataFrame(out).to_csv(args.output/'cat_sensitivity.csv',index=False)
