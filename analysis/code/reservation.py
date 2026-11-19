"""Remove the inherited feasibility-panel restriction and rebuild both tasks for a sensitivity check."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
from chemical_features import fingerprint,chemical_features
repo,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(repo/'src'))
from pilot import stable_seed,learn,predict
from holdout import evaluate_assay,METRICS
from sklearn.model_selection import GroupKFold
out.mkdir(parents=True,exist_ok=True);data=pd.read_csv(repo/'data/record_audit.csv.gz',low_memory=False)
keep=data.loc[data.passed_record_filters].copy()
limits=keep.groupby(['target','smiles']).pic50.transform(lambda x:x.max()-x.min())
keep=keep[limits<=1].copy()
def family(sm):
    m=Chem.MolFromSmiles(sm); Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m,isomericSmiles=False)
keep['family']=keep.smiles.map({sm:family(sm) for sm in keep.smiles.unique()})
keep=keep[keep.groupby(['target','family']).doc.transform('nunique')==1]
keys=['target','doc','assay','family','smiles']; table=keep.groupby(keys,as_index=False).pic50.median().sort_values(keys).drop_duplicates(keys[:-1])
with np.load(repo/'data/features.npz') as z:
    old=list(z['smiles']); missing=sorted(set(table.smiles)-set(old)); fp=np.vstack([z['fp'],[fingerprint(s) for s in missing]]); ds=np.vstack([z['desc'],[chemical_features(s) for s in missing]])
    table['feature_id']=pd.Index(old+missing).get_indexer(table.smiles)
xx=np.column_stack([fp,ds]).astype('float32');res=[];ww=[]
for target,g in table.groupby('target'):
    x=xx[g.feature_id]; y=g.pic50.to_numpy(); mask=g.doc.map(lambda s:stable_seed(s)%5==0); tr=np.flatnonzero(~mask);te=np.flatnonzero(mask)
    assert set(g.iloc[tr].family).isdisjoint(g.iloc[te].family)
    est=np.full(len(tr),np.nan)
    for t,v in GroupKFold(3).split(tr,groups=g.iloc[tr].doc): est[v]=predict(tr[t],tr[v],x,y)
    w,_=learn(g.iloc[tr].assign(pred=est),fp)
    ww.append([target,len(g),len(tr),w])
    for (doc,assay),aa in g.iloc[te].assign(pred=predict(tr,te,x,y)).groupby(['doc','assay']):
        if len(aa)>=10:res.extend([target,doc,assay,*row] for row in evaluate_assay(aa,fp,w) if row[0]=='primary')
frame=pd.DataFrame(res,columns=['target','doc','assay','scope','k','repeat','model','n_queries',*METRICS])
docs=frame.groupby(['target','doc','assay','model'])[METRICS].mean().groupby(['target','doc','model']).mean().reset_index()
docs.to_csv(out/'reservation_released_by_document.csv',index=False)
pd.DataFrame(ww,columns=['target','observations','training_observations','w']).to_csv(out/'reservation_released_counts.csv',index=False)
