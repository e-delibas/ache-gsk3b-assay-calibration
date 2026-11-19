"""Publication-held-out three-reference pilot. Run from any directory."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
GRID = np.linspace(0, 1, 21)
MODELS = ['raw_cat', 'reference_mean', 'offset', 'fixed_quarter', 'learned_global', 'learned_conditional']

def stable_seed(value):
    return int(hashlib.sha256(value.encode()).hexdigest(), 16)

def predict(train, query, x, y):
    model = CatBoostRegressor(iterations=220, depth=6, learning_rate=.05, thread_count=4,
                              random_seed=42, verbose=False, allow_writing_files=False)
    model.fit(x[train], y[train])
    return model.predict(x[query])

def episodes(table, fingerprints):
    for (doc, assay), rows in table.groupby(['doc','assay']):
        if len(rows) < 10:
            continue
        rng = np.random.default_rng(stable_seed(assay) % 2**32)
        p, y = rows.pred.to_numpy(), rows.pic50.to_numpy()
        for repeat in range(20):
            perm = rng.permutation(len(rows))
            k, q = perm[:3], perm[3:]
            ref = fingerprints[rows.feature_id.to_numpy()[k]].astype(bool)
            sims = [np.sum(ref[i]&ref[j])/max(1,np.sum(ref[i]|ref[j])) for i,j in [(0,1),(0,2),(1,2)]]
            agreement = np.sum((p[k]-p[k].mean())*(y[k]-y[k].mean()))
            group = 2*int(np.mean(sims)>=.4) + int(agreement>0)
            yield doc, assay, repeat, group, y[k].mean(), p[q]-p[k].mean(), y[q], p[q]

def learn(table, fp):
    curves = []
    for doc, assay, _, group, center, delta, y, _ in episodes(table,fp):
        losses = np.abs(center+GRID[:,None]*delta-y).mean(axis=1)
        curves.append([doc,assay,group,*losses])
    cols = list(range(len(GRID)))
    frame = pd.DataFrame(curves,columns=['doc','assay','group',*cols])
    # Publication-balanced loss. All repeats of a given assay together have unit weight.
    docs = frame.groupby(['doc','assay'])[cols].mean().groupby('doc').mean()
    pooled = docs.mean().to_numpy()
    global_weight = GRID[pooled.argmin()]
    weights = []
    for group in range(4):
        selected = frame[frame.group==group]
        if selected.empty:
            weights.append(global_weight)
            continue
        subgroup = selected.groupby(['doc','assay'])[cols].mean().groupby('doc').mean()
        weights.append(GRID[(subgroup.sum().to_numpy()+10*pooled).argmin()])
    return global_weight, weights

def score_target(target, data, features, fp):
    data = data.reset_index(drop=True)
    x = features[data.feature_id]
    y = data.pic50.to_numpy()
    records, params, split_rows = [], [], []
    for fold, (train, test) in enumerate(GroupKFold(4).split(x,groups=data.doc)):
        print(target,'fold',fold+1,flush=True)
        assert not set(data.iloc[train].family) & set(data.iloc[test].family)
        train_data = data.iloc[train].copy()
        oof = np.full(len(train),np.nan)
        for t,v in GroupKFold(3).split(train,groups=train_data.doc):
            oof[v] = predict(train[t],train[v],x,y)
        assert np.isfinite(oof).all()
        global_w, local_w = learn(train_data.assign(pred=oof),fp)
        params.append(dict(target=target,fold=fold,global_w=global_w,**{f'group_{i}':w for i,w in enumerate(local_w)}))
        test_data = data.iloc[test].assign(pred=predict(train,test,x,y))
        split_rows.extend((target,doc,fold) for doc in test_data.doc.unique())
        for doc,assay,r,g,center,delta,truth,raw in episodes(test_data,fp):
            weights = [None,0.,1.,.25,global_w,local_w[g]]
            # All positive weights induce the same ranks. A constant has undefined Spearman.
            rho = np.nan
            if np.ptp(raw)>0 and np.ptp(truth)>0:
                rho = np.corrcoef(rankdata(raw),rankdata(truth))[0,1]
            for name,w in zip(MODELS,weights):
                pred = raw if w is None else center+w*delta
                records.append((target,doc,assay,fold,r,name,len(truth),w,np.abs(pred-truth).mean(),
                               rho if w is None or w>0 else np.nan))
    return records,params,split_rows

def main():
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    hashes = json.loads((ROOT/'data/provenance.json').read_text())['snapshot_hashes']
    for name,expected in hashes.items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Veri sürümü farklı: '+name)
    d = pd.read_csv(ROOT/'data/measurements.csv')
    d = d[d.doc.map(lambda doc:stable_seed(doc)%5!=0)].copy()
    with np.load(ROOT/'data/features.npz') as cache:
        fp = cache['fp']
        features = np.column_stack([fp,cache['desc']]).astype(np.float32)
        d['feature_id'] = pd.Index(cache['smiles']).get_indexer(d.smiles)
    assert (d.feature_id>=0).all()
    episodes_out, params, splits = [], [], []
    for target,data in d.groupby('target'):
        rec,par,split = score_target(target,data,features,fp)
        episodes_out.extend(rec)
        params.extend(par)
        splits.extend(split)
    scores = pd.DataFrame(episodes_out,columns=['target','doc','assay','fold','repeat','model','query_n','weight','mae','spearman'])
    scores.to_csv(out/'episodes.csv.gz',index=False)
    pd.DataFrame(params).to_csv(out/'weights.csv',index=False)
    pd.DataFrame(splits,columns=['target','doc','fold']).to_csv(out/'folds.csv',index=False)
    docs = scores.groupby(['target','model','doc','assay'])[['mae','spearman']].mean().groupby(['target','model','doc']).mean()
    summary = docs.groupby(['target','model']).mean().reset_index()
    summary = summary.merge(scores.groupby(['target','model']).agg(assays=('assay','nunique'),documents=('doc','nunique'),
                            defined_rank_fraction=('spearman',lambda v:v.notna().mean())).reset_index())
    summary.to_csv(out/'metrics.csv',index=False)
    contrasts = []
    for target in d.target.unique():
        wide = docs.loc[target].mae.unstack('model')
        rng = np.random.default_rng(42)
        indices = rng.integers(len(wide),size=(2000,len(wide)))
        for model in MODELS[:-1]:
            gain = (wide[model]-wide.learned_conditional).to_numpy()
            lo,hi = np.quantile(gain[indices].mean(axis=1),[.025,.975])
            contrasts.append((target,model,gain.mean(),lo,hi))
    contrasts = pd.DataFrame(contrasts,columns=['target','baseline','mae_gain','low','high'])
    contrasts.to_csv(out/'contrasts.csv',index=False)
    go = (contrasts.query("baseline=='learned_global'").mae_gain>=.02).all() and (contrasts.query("baseline=='reference_mean'").mae_gain>0).all()
    (out/'decision.txt').write_text('GO' if go else 'STOP_CONDITIONAL_WEIGHT')
    print(summary.round(4).to_string(index=False),flush=True)

if __name__=='__main__':
    main()
