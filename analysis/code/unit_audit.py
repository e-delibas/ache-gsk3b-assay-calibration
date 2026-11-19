"""Check Zhao et al. Table 1 against ChEMBL and evaluate the source-verified sensitivity panel.
Arguments: original analysis repository, output directory.
Use RDKit 2026.09.1 for feature generation. No source file is modified.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
repo,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(repo/'src'))
inputs=Path(__file__).resolve().parents[1]/'inputs'
from pilot import stable_seed, predict
from holdout import evaluate_assay,METRICS
from chemical_features import fingerprint,chemical_features
source_pM=dict(zip(['Donepezil','6','7a','7b','7c','7d',*['10'+c for c in 'abcdefghijklmn']],
                   [3460,1360000,1320000,914000,791000,926000,510,28.36,26.21,18.93,13.23,14.15,14.41,10.95,20.94,13.52,29.38,16.72,25.92,21.58]))
assay='CHEMBL4192783';doc='CHEMBL4190278';target='AChE'
records=pd.read_csv(repo/'data/record_audit.csv.gz',low_memory=False).query("target=='AChE'")
allrec=records.rename(columns={'assay':'Assay ChEMBL ID','doc':'Document ChEMBL ID','smiles':'standard_smiles','standard_units':'Standard Units','standard_value':'Standard Value','passed_record_filters':'strict_keep','export_row':'source_row'}).merge(pd.read_csv(inputs/'export_metadata_20.csv'),on='source_row',how='left',validate='one_to_one')
rows=allrec[allrec['Assay ChEMBL ID'].eq(assay)].copy()
assert len(rows)==len(source_pM) and set(rows['Compound Key'])==set(source_pM)
rows['source_pM']=rows['Compound Key'].map(source_pM)
rows['checked_nM']=rows.source_pM/1000
literal=rows['Standard Units'].map({'nM':1,"10'3pM":1,"10'5pM":100,"10'6pM":1000})
np.testing.assert_allclose(rows.checked_nM,literal*rows['Standard Value'])
add=rows[~rows.strict_keep].copy()
assert set(add.all_exclusion_reasons)=={'not_exact_positive_nM;not_standard_valid'}
assert (add.standard_flag_api==0).all() and len(add)==6
reserved=pd.read_csv(inputs/'reserved_81.csv')
recordpool=allrec[allrec.strict_keep|allrec.index.isin(add.index)].copy()
recordpool.loc[add.index,'pic50']=9-np.log10(add.checked_nM)
def family(sm):
    m=Chem.MolFromSmiles(sm);Chem.RemoveStereochemistry(m);return Chem.MolToSmiles(m,isomericSmiles=False)
unique=recordpool.standard_smiles.unique()
family_map={sm:family(sm) for sm in unique}
limits=recordpool.groupby('standard_smiles').pic50.agg(['max','min'])
ok=limits.index[(limits['max']-limits['min']<=1)&~limits.index.isin(reserved.standard_smiles)]
for sm in list(ok):
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(smiles=sm,includeChirality=False) or '__ACYCLIC__'
    if scaffold in set(reserved.scaffold_group) or family_map[sm] in set(reserved.stereo_family):ok=ok.drop(sm,errors='ignore')
pool=recordpool[recordpool.standard_smiles.isin(ok)].copy()
pool['family']=pool.standard_smiles.map(family_map)
pool=pool[pool.groupby('family')['Document ChEMBL ID'].transform('nunique')==1]
keys=['doc','assay','family','smiles']
pool=pool.rename(columns={'Document ChEMBL ID':'doc','Assay ChEMBL ID':'assay','standard_smiles':'smiles'})
rebuilt=pool.groupby(keys,as_index=False).pic50.median().sort_values(keys).drop_duplicates(keys[:-1])
m=pd.read_csv(repo/'data/measurements.csv').query("target=='AChE'").reset_index(drop=True)
def normalized(f):return f[keys+['pic50']].sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(normalized(m[m.assay!=assay]),normalized(rebuilt[rebuilt.assay!=assay]),atol=1e-12,rtol=0)
panel=rebuilt[rebuilt.assay==assay].sort_values(keys).copy()
rows['primary_retained']=rows.standard_smiles.isin(m.loc[m.assay==assay,'smiles'])
rows['augmented_retained']=rows.standard_smiles.isin(panel.smiles)
rows[['Compound Key','source_row','Standard Value','Standard Units','source_pM','checked_nM','standard_flag_api','all_exclusion_reasons','primary_retained','augmented_retained']].to_csv(out/'unit_audit_20.csv',index=False)
with np.load(repo/'data/features.npz') as z:
    fp=z['fp'];features=np.column_stack([fp,z['desc']]).astype('float32');lookup=pd.Index(z['smiles']); ids=lookup.get_indexer(m.smiles)
    new=[sm for sm in panel.smiles if sm not in lookup]
    extra=np.stack([fingerprint(sm) for sm in new]); fp=np.concatenate([fp,extra])
    features=np.vstack([features,np.column_stack([extra,[chemical_features(sm) for sm in new]])]).astype('float32')
    lookup=lookup.append(pd.Index(new))
assert (ids>=0).all()
train=np.flatnonzero(m.doc.map(lambda s:stable_seed(s)%5!=0))
def episodes(panel):
    panel=panel.copy();panel['feature_id']=lookup.get_indexer(panel.smiles)
    x=np.concatenate([features[ids],features[panel.feature_id]])
    y=np.r_[m.pic50,np.full(len(panel),np.nan)]
    panel['pred']=predict(train,np.arange(len(m),len(x)),x,y)
    return pd.DataFrame(evaluate_assay(panel,fp,.75),columns=['scope','k','repeat','model','n_queries',*METRICS])
f=repo/'results/holdout';base=pd.read_csv(f/'episodes.csv.gz')
original=episodes(m[m.assay==assay])
old=base[base.assay==assay];np.testing.assert_allclose(old[METRICS],original[METRICS],equal_nan=True,atol=1e-9)
newrows=episodes(panel).assign(target=target,assay=assay,doc=doc)
full=pd.concat([base[base.assay!=assay],newrows],ignore_index=True)
docs=full.groupby(['target','scope','k','model','doc','assay'])[METRICS].mean().groupby(['target','scope','k','model','doc']).mean().reset_index()
docs.to_csv(out/'augmented_by_document.csv',index=False)
print(rows[['Compound Key','primary_retained','augmented_retained']].to_string(index=False))
print('n',len(panel),'weights fixed; train unchanged')
