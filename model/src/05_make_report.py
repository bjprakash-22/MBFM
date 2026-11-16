import json,pandas as pd
from common import OUT
q=json.loads((OUT/"dataset_quality_summary.json").read_text()) if (OUT/"dataset_quality_summary.json").exists() else {}
parts=["# Computational Results Summary","","## Dataset"]
parts += [f"- {k}: {v}" for k,v in q.items()]
for title,file in [("Accounting role distribution","accounting_role_distribution.csv"),("Diagnostic comparison","diagnostic_model_comparison.csv"),("Clustering metrics","clustering_metrics.csv")]:
    parts += ["",f"## {title}",""]
    p=OUT/file
    if p.exists(): parts.append(pd.read_csv(p).to_markdown(index=False))
parts += ["","## Limitations","- Three diagnostic pairs are illustrative, not a statistically valid benchmark.","- Similarity is not accounting equivalence.","- Ontology labels are not independent ground truth.","- High missingness must not be treated as zero values.","- Clustering is exploratory unless independently validated.","- Cross-sectional data does not establish forecasting performance.","- Independent expert annotation and stronger baselines are needed for a publishable evaluation."]
(OUT/"research_results_summary.md").write_text("\\n".join(parts),encoding="utf-8")
print("Created outputs/research_results_summary.md")
