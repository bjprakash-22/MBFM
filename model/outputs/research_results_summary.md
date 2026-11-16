# Computational Results Summary\n\n## Dataset\n- companies: 13162\n- financial_variables: 182\n- total_possible_values: 2395484\n- available_values: 135902\n- overall_coverage_percent: 5.67\n- overall_missingness_percent: 94.33\n- zero_observation_variables: 3\n- median_features_per_company: 0.0\n\n## Accounting role distribution\n\n| Accounting_Role   |   Concept_Count |
|:------------------|----------------:|
| Expense           |              84 |
| Income            |              38 |
| Other             |               7 |
| Adjustment        |               6 |
| Inventory         |               5 |
| Gain              |               1 |\n\n## Diagnostic comparison\n\n| Concept_A                  | Concept_B                         |   Expected_Label |   Original_Hybrid |   Role_Aware | Role_A   | Role_B    | Status   |
|:---------------------------|:----------------------------------|-----------------:|------------------:|-------------:|:---------|:----------|:---------|
| Employee benefits expenses | Staff welfare & training expenses |                1 |            0.8458 |       0.8921 | Expense  | Expense   | Found    |
| Interest income from banks | Interest expense                  |                0 |            0.4622 |       0.3535 | Income   | Expense   | Found    |
| Sales of goods             | Purchase of finished goods        |                0 |            0.499  |       0.3793 | Income   | Inventory | Found    |\n\n## Clustering metrics\n\n| Embedding   |   K |   Silhouette |   Davies_Bouldin |
|:------------|----:|-------------:|-----------------:|
| SBERT       |   2 |      0.06481 |          3.54704 |
| SBERT       |   3 |      0.07078 |          3.2606  |
| SBERT       |   4 |      0.05568 |          3.26345 |
| SBERT       |   5 |      0.06348 |          3.00063 |
| SBERT       |   6 |      0.0719  |          2.83229 |
| SBERT       |   7 |      0.0743  |          2.70441 |
| SBERT       |   8 |      0.0811  |          2.58134 |
| SBERT       |   9 |      0.08358 |          2.47646 |
| SBERT       |  10 |      0.08446 |          2.4628  |
| SBERT       |  11 |      0.08071 |          2.39604 |
| SBERT       |  12 |      0.0797  |          2.43395 |
| FinBERT     |   2 |      0.08032 |          3.3029  |
| FinBERT     |   3 |      0.08382 |          2.80441 |
| FinBERT     |   4 |      0.05192 |          3.01513 |
| FinBERT     |   5 |      0.03396 |          2.87864 |
| FinBERT     |   6 |      0.05825 |          2.81676 |
| FinBERT     |   7 |      0.0461  |          2.67644 |
| FinBERT     |   8 |      0.05275 |          2.70361 |
| FinBERT     |   9 |      0.05048 |          2.62075 |
| FinBERT     |  10 |      0.05528 |          2.56043 |
| FinBERT     |  11 |      0.05927 |          2.41837 |
| FinBERT     |  12 |      0.05001 |          2.53836 |
| Combined    |   2 |      0.05678 |          3.82931 |
| Combined    |   3 |      0.06034 |          3.51693 |
| Combined    |   4 |      0.04218 |          3.43222 |
| Combined    |   5 |      0.04802 |          3.45773 |
| Combined    |   6 |      0.05825 |          3.03712 |
| Combined    |   7 |      0.0554  |          2.95437 |
| Combined    |   8 |      0.0614  |          2.81088 |
| Combined    |   9 |      0.05723 |          2.71198 |
| Combined    |  10 |      0.07515 |          2.55709 |
| Combined    |  11 |      0.06428 |          2.64098 |
| Combined    |  12 |      0.06566 |          2.61334 |\n\n## Limitations\n- Three diagnostic pairs are illustrative, not a statistically valid benchmark.\n- Similarity is not accounting equivalence.\n- Ontology labels are not independent ground truth.\n- High missingness must not be treated as zero values.\n- Clustering is exploratory unless independently validated.\n- Cross-sectional data does not establish forecasting performance.\n- Independent expert annotation and stronger baselines are needed for a publishable evaluation.