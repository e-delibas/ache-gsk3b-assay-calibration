"""Produce paired document-bootstrap confidence intervals for Table 4 / Table S11.

Uses stored 20-reference-draw, assay-averaged and document-averaged results,
not retraining. Each source document is one independent bootstrap unit.

Run from repository root:
python analysis/code/comparator_intervals.py --repo . --output analysis/outputs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = [
    ('train_mean', 'Training mean'),
    ('mean', 'Reference mean'),
    ('ref_median', 'Reference median'),
    ('shrinkage', 'Shrinkage to training mean'),
    ('raw', 'Raw CatBoost'),
    ('calibrated', 'Calibrated CatBoost'),
    ('nearest_reference', 'Nearest reference'),
    ('oracle_query_mean', 'Oracle query mean'),
    ('oracle_query_median', 'Oracle query median'),
]
TARGETS = [('AChE', 31), ('GSK3B', 10)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('analysis/outputs'))
    p.add_argument('--draws',type=int,default=50_000)
    p.add_argument('--seed',type=int,default=20_261_120)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    by=pd.read_csv(a.repo/'analysis/outputs/primary_by_document.csv')
    summary=pd.read_csv(a.repo/'analysis/outputs/primary_summary.csv')
    source=pd.read_csv(a.repo/'results/holdout/by_document.csv')
    docrows=[]; rows=[]
    for k,(target,n_expect) in enumerate(TARGETS):
        v=by.loc[by.target.eq(target)].pivot(index='doc',columns='method',values='mae')
        assert len(v)==n_expect and v.notna().all().all(),f'{target} docs'
        assert set(v)=={code for code,name in METHODS}
        # The three shared baselines reproduce the original held-out analysis.
        orig=source[(source.target==target)&(source.scope=='primary')&(source.k==3)]
        w=orig.pivot(index='doc',columns='model',values='mae')
        for col in ['raw','mean','calibrated','nearest_reference']:
            np.testing.assert_allclose(v.loc[w.index,col],w[col],rtol=0,atol=1e-10)
        for code,_ in METHODS:
            stored=float(summary[(summary.target==target)&(summary.method==code)].mae.iloc[0])
            assert np.isclose(v[code].mean(),stored,atol=1e-11), (target,code)
        # Reuse one set of document draws across methods; paired within each document.
        rng=np.random.default_rng(a.seed+100*k)
        ix=rng.integers(0,len(v),size=(a.draws,len(v)))
        for code,name in METHODS:
            delta=(v[code]-v['calibrated']).to_numpy()
            mean=float(delta.mean())
            low,high=np.quantile(delta[ix].mean(axis=1),[.025,.975])
            wins=int((delta>1e-12).sum());losses=int((delta< -1e-12).sum())
            ties=len(delta)-wins-losses
            rows.append(dict(target=target,method=code,comparator=name,n_documents=len(v),
                comparator_mae=float(v[code].mean()),calibrated_mae=float(v.calibrated.mean()),
                delta_mae=mean,ci_low=float(low),ci_high=float(high),
                calibrated_better_docs=wins,comparator_better_docs=losses,tied_docs=ties,
                bootstrap_replicates=a.draws,bootstrap_seed=a.seed+100*k))
            for doc,value,cal,diff in zip(v.index,v[code],v.calibrated,delta):
                docrows.append(dict(target=target,doc=doc,method=code,comparator=name,
                        comparator_mae=float(value),calibrated_mae=float(cal),delta_mae=float(diff)))
    f=pd.DataFrame(rows)
    b=pd.DataFrame(docrows)
    assert len(f)==18 and len(b)==369
    assert abs(f.query("target=='GSK3B' and method=='shrinkage'").iloc[0].delta_mae+0.00036)<0.00001
    assert abs(f.query("target=='AChE' and method=='shrinkage'").iloc[0].delta_mae-0.029006)<.00001
    f.to_csv(a.output/'comparator_paired_summary.csv',index=False,float_format='%.12g')
    b.to_csv(a.output/'comparator_paired_by_document.csv',index=False,float_format='%.12g')
    print(f[['target','comparator','comparator_mae','delta_mae','ci_low','ci_high']].to_string(index=False))
    print('PASS: 18 aggregate rows, 369 document/method rows, all original 3-method MAEs reproduced')
if __name__=='__main__': main()
