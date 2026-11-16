import argparse,numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from common import OUT,corpus
p=argparse.ArgumentParser(); p.add_argument("--skip-finbert",action="store_true"); a=p.parse_args()
d=corpus(); s=np.load(OUT/"sbert_raw.npy"); mats={"SBERT":s}
f=OUT/"finbert_raw.npy"
if f.exists() and not a.skip_finbert:
    z=np.load(f); mats["FinBERT"]=z; mats["Combined"]=np.concatenate([s,z],axis=1)
rows=[]; memberships=d[["Concept_ID","Original_Variable","Refined_Domain","Refined_Category"]].copy()
for name,x in mats.items():
    best=None
    for k in range(2,min(12,len(x)-1)+1):
        lab=KMeans(n_clusters=k,random_state=42,n_init=20).fit_predict(x)
        sil=silhouette_score(x,lab); db=davies_bouldin_score(x,lab)
        rows.append({"Embedding":name,"K":k,"Silhouette":round(float(sil),5),"Davies_Bouldin":round(float(db),5)})
        if best is None or sil>best[0]: best=(sil,k,lab)
    memberships[name+"_Cluster"]=best[2]; print(name,"best k",best[1],"silhouette",round(best[0],5))
pd.DataFrame(rows).to_csv(OUT/"clustering_metrics.csv",index=False); memberships.to_csv(OUT/"concept_clusters.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
