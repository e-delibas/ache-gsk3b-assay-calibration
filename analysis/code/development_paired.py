"""Document-level paired inference for original 4x3 nested development evaluation.
Uses only cached out-of-fold predictions/episode losses; no model training.
The unit of resampling/inference is source document (not assay or repeated reference draw).
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
cli = argparse.ArgumentParser(description='Reproduce paired document statistics from nested development episodes.')
cli.add_argument('--repo', type=Path, default=Path('.'), help='Repository root containing results/episodes.csv.gz')
cli.add_argument('--output', type=Path, default=Path('analysis/outputs'), help='Output directory')
args = cli.parse_args()
ROOT=args.repo.expanduser().resolve()
OUT=args.output.expanduser().resolve()
OUT.mkdir(parents=True,exist_ok=True)
rows=pd.read_csv(ROOT/'results/episodes.csv.gz')
# 1: repetitions mean inside assay; 2: equally weight assays inside doc; 3: docs equally weighted
by_doc=(rows.groupby(['target','doc','assay','model'],as_index=False)['mae'].mean()
           .groupby(['target','doc','model'],as_index=False)['mae'].mean())
wide=by_doc.pivot(index=['target','doc'],columns='model',values='mae')
summary=[]; pairs=[]
comparisons=[('reference_mean','learned_global'),('raw_cat','learned_global'),('offset','learned_global'),('reference_mean','learned_conditional')]
for ti,target in enumerate(['AChE','GSK3B']):
 w=wide.loc[target]
 print('\nTARGET',target,'n_docs',len(w))
 for base,model in comparisons:
  delta=(w[base]-w[model]).to_numpy()
  rng=np.random.default_rng(20261120+ti*100)
  # 50,000 paired, document-cluster bootstrap resamples
  B=50000
  ix=rng.integers(0,len(delta),size=(B,len(delta)))
  boot=delta[ix].mean(axis=1)
  lo,hi=np.quantile(boot,[.025,.975])
  mboot=np.median(delta[ix],axis=1)
  medlo,medhi=np.quantile(mboot,[.025,.975])
  win=int((delta>1e-10).sum());loss=int((delta< -1e-10).sum());tie=len(delta)-win-loss
  p=binomtest(win,win+loss,.5,alternative='two-sided').pvalue if win+loss else np.nan
  r=dict(target=target,baseline=base,model=model,n_documents=len(w),mean_gain=delta.mean(),median_gain=np.median(delta),
         mean_ci_low=lo,mean_ci_high=hi,median_ci_low=medlo,median_ci_high=medhi,improved=win,worsened=loss,ties=tie,sign_test_p=p,
         baseline_mae=w[base].mean(),model_mae=w[model].mean(),positive_pct=100*win/(win+loss))
  summary.append(r)
  print(pd.Series(r).to_string())
  for d,value in zip(w.index,delta): pairs.append(dict(target=target,doc=d,baseline=base,model=model,mae_gain=value))
pd.DataFrame(summary).to_csv(OUT/'nested_development_paired_summary.csv',index=False)
pd.DataFrame(pairs).to_csv(OUT/'nested_development_paired_by_document.csv',index=False)
# verify joins and scoring
assert wide.isna().sum().sum()==0,wide.isna().sum()
ref=pd.read_csv(ROOT/'results/metrics.csv')
for _,z in ref.iterrows():
 assert abs(wide.loc[z.target,z.model].mean()-z.mae)<1e-10,(z.target,z.model)
print('All model MAEs exactly match deposited results/metrics.csv')
