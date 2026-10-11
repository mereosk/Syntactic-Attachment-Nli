import pandas as pd, sys
from pathlib import Path
HUMAN_DIR=Path(__file__).resolve().parents[1]/"data"/"human_results"

PROF="How would you describe your English language proficiency?"
for src,dst in [("human_study_items - Form responses 1.csv","v1.csv"),("human_study_items - Form responses 2.csv","v2.csv")]:
    src,dst=HUMAN_DIR/src,HUMAN_DIR/dst
    r=pd.read_csv(src)
    r=r[[c for c in r.columns if not c.endswith("[Row 1]") and not c.startswith("Column")]]
    # proficiency: v1 has it in col 3 (renamed) + an empty duplicate at the end; v2 only at the end
    profs=[c for c in r.columns if c.startswith(PROF)]
    r["proficiency"]=r[profs].bfill(axis=1).iloc[:,0]
    r=r.drop(columns=profs+[c for c in r.columns if c.startswith("What is your native")])
    r.columns=[c.rsplit(" [",1)[0] if c.endswith("[Your answer]") else c for c in r.columns]
    r.to_csv(dst,index=False); print(dst, r.shape, r["proficiency"].tolist())
