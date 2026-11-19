import sys
from pathlib import Path
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
repo,out=map(Path,sys.argv[1:3])
m=pd.read_csv(repo/'data/measurements.csv'); lookup={k:MurckoScaffold.MurckoScaffoldSmiles(smiles=k,includeChirality=False) or '__ACYCLIC__' for k in m.smiles.unique()};m['scaffold']=m.smiles.map(lookup)
g=m.groupby(['target','doc','assay']).agg(scaffolds=('scaffold','nunique'),n=('pic50','size'),sd=('pic50','std'),range_pic50=('pic50',lambda x:x.max()-x.min())).reset_index()
base=pd.read_csv(out/'document_characteristics.csv')
v=base.merge(g[['target','doc','assay','scaffolds']],on=['target','doc','assay'],validate='one_to_one')
v.to_csv(out/'document_scaffolds_and_performance.csv',index=False)
rows=[]
for target,sub in v.groupby('target'):
 for col in ['n','sd','potency_range','scaffolds','raw_rho']:
    rows.append([target,col,spearmanr(sub.gain_over_mean,sub[col]).statistic])
pd.DataFrame(rows,columns=['target','characteristic','spearman_with_gain']).to_csv(out/'gain_correlations.csv',index=False)
g['size_group']=pd.cut(g.n,bins=[0,5,9,float('inf')],labels=['1–5','6–9','10+'])
g.groupby(['target','size_group'],observed=True).agg(assays=('assay','size'),median_size=('n','median'),median_within_SD=('sd','median'),median_range=('range_pic50','median'),median_scaffolds=('scaffolds','median')).to_csv(out/'assay_size_distributions.csv')
