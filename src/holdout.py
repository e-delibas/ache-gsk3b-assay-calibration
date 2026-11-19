"""Reproduce the held-out three-reference result and fixed-budget sensitivity checks."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from pilot import ROOT, stable_seed, predict, learn

OUT = ROOT / 'results' / 'holdout'
METRICS = ['mae','mse','level_mse','within_mse','rho','rank_defined']

def measure(truth, prediction):
    residual = prediction-truth
    level = residual.mean()**2
    within = residual.var()
    mse = np.mean(residual**2)
    assert np.isclose(mse,level+within)
    defined = np.ptp(prediction)>0 and np.ptp(truth)>0
    rho = np.corrcoef(rankdata(truth),rankdata(prediction))[0,1] if defined else np.nan
    return [np.abs(residual).mean(),mse,level,within,rho,float(defined)]

def evaluate_assay(rows, fp, weight):
    y = rows.pic50.to_numpy()
    pred = rows.pred.to_numpy()
    bits = fp[rows.feature_id].astype(float)
    intersections = bits @ bits.T
    counts = bits.sum(axis=1)
    sim = intersections / np.maximum(1,counts[:,None]+counts[None,:]-intersections)
    rng = np.random.default_rng(stable_seed(rows.assay.iloc[0]) % 2**32)
    for repeat in range(20):
        perm = rng.permutation(len(y))
        for scope,k,q in [('primary',3,perm[3:]), *[('common_query',n,perm[5:]) for n in (1,3,5)]]:
            ref = perm[:k]
            mu = y[ref].mean()
            forecasts = {'raw':pred[q], 'mean':np.repeat(mu,len(q)),
                         'calibrated':mu+weight*(pred[q]-pred[ref].mean())}
            if scope=='primary':
                distances = sim[np.ix_(q,ref)]
                nearest = distances==distances.max(axis=1,keepdims=True)
                forecasts['nearest_reference'] = (nearest @ y[ref])/nearest.sum(axis=1)
            for name, forecast in forecasts.items():
                yield [scope,k,repeat,name,len(q),*measure(y[q],forecast)]

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    for f, digest in json.loads((ROOT/'data/provenance.json').read_text())['snapshot_hashes'].items():
        assert hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest()==digest
    table = pd.read_csv(ROOT/'data/measurements.csv')
    with np.load(ROOT/'data/features.npz') as cache:
        fp = cache['fp']
        features = np.column_stack([fp,cache['desc']]).astype('float32')
        table['feature_id'] = pd.Index(cache['smiles']).get_indexer(table.smiles)
    assert (table.feature_id>=0).all()
    episodes, fits, manifest = [], [], []
    for target, data in table.groupby('target'):
        x = features[data.feature_id]
        y = data.pic50.to_numpy()
        hidden = data.doc.map(lambda s:stable_seed(s)%5==0)
        tr,te = np.flatnonzero(~hidden),np.flatnonzero(hidden)
        assert set(data.iloc[tr].doc).isdisjoint(data.iloc[te].doc)
        assert set(data.iloc[tr].family).isdisjoint(data.iloc[te].family)
        oof = np.full(len(tr),np.nan)
        for t,v in GroupKFold(3).split(tr,groups=data.doc.iloc[tr]):
            oof[v] = predict(tr[t],tr[v],x,y)
        assert np.isfinite(oof).all()
        weight,_ = learn(data.iloc[tr].assign(pred=oof),fp)
        fits.append([target,weight])
        tested = data.iloc[te].assign(pred=predict(tr,te,x,y))
        for (doc,assay), rows in tested.groupby(['doc','assay']):
            manifest.append([target,doc,assay,len(rows),len(rows)>=10])
            if len(rows)>=10:
                episodes.extend([target,doc,assay,*row] for row in evaluate_assay(rows,fp,weight))
    pd.DataFrame(fits,columns=['target','weight']).to_csv(OUT/'weights.csv',index=False)
    pd.DataFrame(manifest,columns=['target','doc','assay','n_families','scored']).to_csv(OUT/'holdout_manifest.csv',index=False)
    scores = pd.DataFrame(episodes,columns=['target','doc','assay','scope','k','repeat','model','n_queries',*METRICS])
    scores.to_csv(OUT/'episodes.csv.gz',index=False)
    keys = ['target','scope','k','model']
    docs = scores.groupby(keys+['doc','assay'])[METRICS].mean().groupby(keys+['doc']).mean()
    docs.to_csv(OUT/'by_document.csv')
    summary = docs.groupby(keys).mean()
    summary.to_csv(OUT/'summary.csv')
    contrasts = []
    for target in table.target.unique():
        wide = docs.xs((target,'primary',3),level=['target','scope','k']).mae.unstack('model')
        ix = np.random.default_rng(42).integers(len(wide),size=(2000,len(wide)))
        for b in ['mean','raw','nearest_reference']:
            diff = (wide[b]-wide.calibrated).to_numpy()
            lo,hi=np.quantile(diff[ix].mean(axis=1),[.025,.975])
            contrasts.append([target,b,len(diff),diff.mean(),lo,hi])
    pd.DataFrame(contrasts,columns=['target','baseline','n_docs','gain','low','high']).to_csv(OUT/'paired_ci.csv',index=False)
    print(summary[['mae','rho']].round(4).to_string())
if __name__=='__main__':
    run()
