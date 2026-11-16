import argparse,re,numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from common import OUT,FIG,corpus
p=argparse.ArgumentParser(); p.add_argument("--skip-finbert",action="store_true"); a=p.parse_args()
d=corpus().reset_index(drop=True); sbert=cosine_similarity(np.load(OUT/"sbert_raw.npy"))
fp=OUT/"finbert_raw.npy"
fin=cosine_similarity(np.load(fp)) if fp.exists() and not a.skip_finbert else None
sem=.5*sbert+.5*fin if fin is not None else sbert
dom=d.Refined_Domain.astype(str).values; cat=d.Refined_Category.astype(str).values; qua=d.Refined_Qualifier.replace("","Unknown").astype(str).values
ds=np.equal.outer(dom,dom).astype(float); cs=np.equal.outer(cat,cat).astype(float); qs=np.equal.outer(qua,qua).astype(float)
hy=.5*sem+.2*ds+.2*cs+.1*qs; np.fill_diagonal(hy,1)
def role(r):
    t=str(r.Original_Variable).lower(); dm=str(r.Refined_Domain).lower()
    if dm=="inventory" or re.search(r"\\binventory\\b|change in stock|finished goods",t): return "Inventory"
    if dm=="accounting adjustments" or re.search(r"less:|capitalised|capitalized|tax credits|deductions",t): return "Adjustment"
    if dm=="gains and losses":
        if re.search(r"\\bloss\\b|losses|impairment",t): return "Loss"
        if re.search(r"\\bgain\\b|gains|profit",t): return "Gain"
        return "Other"
    if dm=="expenses": return "Expense"
    if dm=="revenue":
        if re.search(r"\\bloss\\b|losses",t): return "Loss"
        if re.search(r"\\bgain\\b|gains",t): return "Gain"
        return "Income"
    if re.search(r"\\bloss\\b|losses|impairment",t): return "Loss"
    if re.search(r"\\bgain\\b|gains",t): return "Gain"
    if re.search(r"\\bincome\\b|revenue|sales|dividend",t): return "Income"
    if re.search(r"expense|expenditure|charges|cost|fees|salaries|wages|paid|consumed|borrowings",t): return "Expense"
    return "Other"
d["Accounting_Role"]=d.apply(role,axis=1); rv=d.Accounting_Role.values; rs=np.equal.outer(rv,rv).astype(float)
ra=.35*sem+.15*ds+.15*cs+.1*qs+.25*rs; np.fill_diagonal(ra,1)
np.save(OUT/"ontology_aware_hybrid_similarity.npy",hy); np.save(OUT/"accounting_role_aware_similarity.npy",ra); d.to_csv(OUT/"concepts_with_accounting_roles.csv",index=False)
d.Accounting_Role.value_counts().rename_axis("Accounting_Role").reset_index(name="Concept_Count").to_csv(OUT/"accounting_role_distribution.csv",index=False)
def find(q):
    ix=np.flatnonzero(d.Original_Variable.str.contains(q,case=False,regex=False,na=False).values); return int(ix[0]) if len(ix) else None
pairs=[("Employee benefits","Staff welfare",1),("Interest income from banks","Interest expense",0),("Sales of goods","Purchase of finished goods",0)]
rows=[]
for x,y,label in pairs:
    i,j=find(x),find(y)
    if i is None or j is None: rows.append({"A":x,"B":y,"Expected_Label":label,"Status":"Not found"}); continue
    rows.append({"Concept_A":d.loc[i,"Original_Variable"],"Concept_B":d.loc[j,"Original_Variable"],"Expected_Label":label,"Original_Hybrid":round(float(hy[i,j]),4),"Role_Aware":round(float(ra[i,j]),4),"Role_A":rv[i],"Role_B":rv[j],"Status":"Found"})
pd.DataFrame(rows).to_csv(OUT/"diagnostic_model_comparison.csv",index=False)
plt.figure(figsize=(8,7)); plt.imshow(ra,vmin=0,vmax=1,aspect="auto"); plt.colorbar(label="Similarity"); plt.title("Accounting-role-aware similarity"); plt.tight_layout(); plt.savefig(FIG/"role_aware_similarity.png",dpi=180); plt.close()
print("ROLE DISTRIBUTION"); print(d.Accounting_Role.value_counts().to_string())
print("\\nDIAGNOSTIC PAIRS"); print(pd.DataFrame(rows).to_string(index=False))
