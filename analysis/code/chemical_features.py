import numpy as np
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

DESCRIPTORS=['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds',
 'RingCount','NumAromaticRings','NumAliphaticRings','FractionCSP3','HeavyAtomCount',
 'NHOHCount','NOCount','NumHeteroatoms','LabuteASA','BalabanJ','BertzCT','HallKierAlpha',
 'MaxPartialCharge','MinPartialCharge']

def fingerprint(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: raise ValueError('Invalid standardized SMILES: '+smiles)
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(m)
    arr=np.zeros(2048,dtype=np.uint8);DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

def chemical_features(smiles):
    m=Chem.MolFromSmiles(smiles);values=[]
    for name in DESCRIPTORS:
        try:v=float(getattr(Descriptors,name)(m))
        except Exception:v=np.nan
        values.append(v if np.isfinite(v) else np.nan)
    return values

