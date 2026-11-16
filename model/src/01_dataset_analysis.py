import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import OUT,FIG,financial,savej
d=financial()
ids=[c for c in ["Company_Key","Company Name"] if c in d]
cols=[c for c in d if c not in ids]
x=d[cols].copy()
for c in cols: x[c]=pd.to_numeric(x[c].astype(str).str.replace(",","",regex=False).str.strip(),errors="coerce")
fc=pd.DataFrame({"Variable":cols,"Available_Values":x.notna().sum().values,"Missing_Values":x.isna().sum().values,"Coverage_Percent":(x.notna().mean().values*100).round(2)})
fc["Missing_Percent"]=(100-fc.Coverage_Percent).round(2)
fc=fc.sort_values("Coverage_Percent",ascending=False)
cc=pd.DataFrame({"Company_Key":d["Company_Key"] if "Company_Key" in d else np.arange(len(d)),"Company_Name":d["Company Name"] if "Company Name" in d else "","Available_Features":x.notna().sum(axis=1)})
cc["Missing_Features"]=len(cols)-cc.Available_Features
cc["Coverage_Percent"]=(cc.Available_Features/len(cols)*100).round(2)
s={"companies":len(d),"financial_variables":len(cols),"total_possible_values":int(x.size),"available_values":int(x.notna().sum().sum()),"overall_coverage_percent":round(float(x.notna().mean().mean()*100),2),"overall_missingness_percent":round(float(x.isna().mean().mean()*100),2),"zero_observation_variables":int((x.notna().sum()==0).sum()),"median_features_per_company":float(cc.Available_Features.median())}
fc.to_csv(OUT/"financial_feature_coverage.csv",index=False); cc.to_csv(OUT/"financial_company_coverage.csv",index=False); x.to_csv(OUT/"financial_numeric_matrix.csv",index=False); savej(s,OUT/"dataset_quality_summary.json")
plt.figure(figsize=(9,5)); plt.hist(fc.Coverage_Percent,bins=20); plt.xlabel("Variable coverage (%)"); plt.ylabel("Number of variables"); plt.title("Financial variable coverage"); plt.tight_layout(); plt.savefig(FIG/"variable_coverage.png",dpi=180); plt.close()
plt.figure(figsize=(9,5)); plt.hist(cc.Available_Features,bins=25); plt.xlabel("Available variables per company"); plt.ylabel("Number of companies"); plt.title("Company reporting coverage"); plt.tight_layout(); plt.savefig(FIG/"company_coverage.png",dpi=180); plt.close()
print("DATA QUALITY"); [print(k,":",v) for k,v in s.items()]
print("\\nTOP 15 VARIABLES"); print(fc.head(15).to_string(index=False))
