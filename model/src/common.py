from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; OUT=ROOT/"outputs"; FIG=OUT/"figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
FIN=DATA/"unified_consolidated_financial_data.csv"
COR=DATA/"financial_nlp_embedding_corpus.csv"
def inputs():
    missing=[str(p) for p in [FIN,COR] if not p.exists()]
    if missing: raise FileNotFoundError("Copy required notebook output CSVs into data/: "+", ".join(missing))
def corpus():
    inputs(); d=pd.read_csv(COR).fillna("")
    req=["Original_Variable","Normalized_Variable","Refined_Domain","Refined_Category"]
    miss=[c for c in req if c not in d]
    if miss: raise ValueError(f"Embedding corpus missing columns: {miss}")
    for c in ["Refined_Qualifier","Semantic_Description","Concept_ID"]:
        if c not in d: d[c]=""
    if d["Concept_ID"].eq("").all(): d["Concept_ID"]=[f"C{i:04d}" for i in range(len(d))]
    return d
def financial():
    inputs(); return pd.read_csv(FIN,low_memory=False)
def savej(x,p):
    with open(p,"w",encoding="utf-8") as f: json.dump(x,f,indent=2,default=str)
