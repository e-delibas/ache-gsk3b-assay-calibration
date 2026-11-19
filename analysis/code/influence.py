"""Document-level influence and common-draw rank contrasts for the archived models."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, bootstrap

arg=argparse.ArgumentParser()
arg.add_argument('repository',type=Path);arg.add_argument('--output',type=Path,default=Path('analysis/outputs'))
args=arg.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
root=args.repository;h=root/'results/holdout'
sources=pd.read_csv(root/'data/sources/heldout_sources.csv').set_index('doc')
bydoc=pd.read_csv(h/'by_document.csv').query("scope=='primary' and k==3")
draw=pd.read_csv(h/'episodes.csv.gz').query("scope=='primary' and k==3")
measure=pd.read_csv(root/'data/measurements.csv')
flag='CHEMBL4190278'
def stats(delta):
    x=np.array(delta);idx=np.random.default_rng(42).integers(len(x),size=(2000,len(x)))
    lo,hi=np.quantile(x[idx].mean(axis=1),[.025,.975])
    bca=bootstrap((x,),np.mean,method="BCa",n_resamples=10000,rng=np.random.default_rng(42)).confidence_interval
    return dict(bca_low=bca.low,bca_high=bca.high,n=len(x),mean_gain=x.mean(),median_gain=np.median(x),low=lo,high=hi,
                wins=(x>0).sum(),losses=(x<0).sum(),sign_p=binomtest((x>0).sum(),(x!=0).sum()).pvalue)
panels=[]; doc_info=[]; ranking=[]; mse=[]
for target,g in bydoc.groupby('target'):
    w=g.pivot(index='doc',columns='model',values='mae')
    gain=w['mean']-w.calibrated
    annotations=sources.loc[w.index]
    known=~annotations.unknown & annotations.doc_type.ne('PATENT')
    subsets={'all':w.index,'four_flags_excluded':annotations[~annotations.domain_issue].index,
             'unknown_and_patent_excluded':annotations[known].index,
             'explicit_human_journal':annotations[~annotations.domain_issue & annotations.explicit_human & annotations.doc_type.ne('PATENT')].index,
             'unknown_patent_and_outlier_excluded':annotations[known & (annotations.index!=flag)].index}
    if target=='AChE':
        subsets['top_two_ref_gains_excluded']=w.index.difference(gain.nlargest(2).index)
        subsets['high_potency_doc_excluded']=w.index.difference([flag])
    for doc in w.index:
        subsets['leave_out_'+doc]=w.index.difference([doc])
    for label,idx in subsets.items():
        for base in ['raw','mean']:
            panels.append(dict(target=target,panel=label,baseline=base,**stats(w.loc[idx,base]-w.loc[idx,'calibrated'])))
        terms=g[g.doc.isin(idx)].groupby('model')[['mae','mse','level_mse','within_mse']].mean()
        share=(terms.level_mse.raw-terms.level_mse.calibrated)/(terms.mse.raw-terms.mse.calibrated)
        for model,line in terms.iterrows():
            mse.append(dict(target=target,panel=label,model=model,raw_to_calibrated_level_share=share,**line.to_dict()))
    for doc in w.index:
        a=annotations.loc[doc]; assay=measure.query('target==@target and doc==@doc')[lambda tab:tab.assay.eq(a.assay)]
        doc_info.append(dict(target=target,doc=doc,assay=a.assay,n=len(assay),sd=assay.pic50.std(),potency_range=assay.pic50.max()-assay.pic50.min(),gain_over_mean=gain[doc],
                             gain_over_raw=w.raw[doc]-w.calibrated[doc],raw_rho=g[(g.doc==doc)&(g.model=='raw')].rho.item()))
    e=draw[draw.target.eq(target)].pivot(index=['doc','assay','repeat'],columns='model',values='rho')[['raw','nearest_reference']].dropna()
    e=e.groupby(['doc','assay']).mean().groupby('doc').mean()
    ranking.append(dict(target=target,rho_raw=e.raw.mean(),rho_nearest=e.nearest_reference.mean(),**stats(e.nearest_reference-e.raw)))
for file,rows in [('influence_and_provenance',panels),('error_components_by_subset',mse),('document_characteristics',doc_info),('spearman_matched',ranking)]:
    pd.DataFrame(rows).to_csv(out/(file+'.csv'),index=False)
