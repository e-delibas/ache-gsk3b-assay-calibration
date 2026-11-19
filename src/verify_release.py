"""Static preflight: checks release files, archived data hashes, and high-level numerical consistency.

This is a consistency test, NOT a new independent refit of all models.
"""
from pathlib import Path
import json, hashlib, csv
import numpy as np
import pandas as pd
base=Path(__file__).resolve().parents[1]
for name,digest in json.loads((base/'data/provenance.json').read_text())['snapshot_hashes'].items():
 assert hashlib.sha256((base/'data'/name).read_bytes()).hexdigest()==digest, name
assert (base/'figures/Figure_3_Error_Decomposition.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
a=pd.read_csv(base/'analysis/outputs/unit_audit_20.csv')
assert len(a)==20 and int((a.augmented_retained & ~a.primary_retained).sum())==5
assert int((a.primary_retained).sum())==14
c=pd.read_csv(base/'analysis/outputs/unit_panel_contrasts.csv')
assert len(c)==8 and set(c.baseline)=={'raw','mean'}
d=pd.read_csv(base/'analysis/outputs/table2_by_panel.csv')
assert len(d)==16
m=pd.read_csv(base/'analysis/outputs/primary_summary.csv')
assert len(m)==18 and m.target.nunique()==2
for target,raw,mean,cal in [('AChE',.943,.536,.507),('GSK3B',.710,.611,.608)]:
 mm=m[m.target==target].set_index('method').mae
 # Strict panel: match article metrics at the displayed three-decimal precision.
 assert abs(mm['raw']-raw)<.0015
 assert abs(mm['mean']-mean)<.0015
 assert abs(mm['calibrated']-cal)<.0015
p=pd.read_csv(base/'analysis/outputs/influence_and_provenance.csv')
assert len(p)==106
for target,n in [('AChE',31),('GSK3B',10)]:
 assert int(p[(p.target==target)&(p.panel=='all')&(p.baseline=='mean')].iloc[0].n)==n
assert (base/'supplementary/SuppInfo_S1_Three_Reference_Calibration.docx').exists()
print('PASS: data hashes, source audit, figure, tables, panel counts, rounded article metrics, Supporting Information')
