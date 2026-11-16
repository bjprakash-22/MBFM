# Accounting-Aware Semantic Alignment of Financial Statement Concepts Using Transformer Embeddings and Ontology-Guided Similarity

layout:
```
model/
  data/
    .csv
  src/
  outputs/
  run_all.py
```

## Windows setup
Open PowerShell in this folder:
```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run everything
```
python run_all.py
```
The first run downloads `all-MiniLM-L6-v2` and `ProsusAI/finbert` from Hugging Face. 
To skip FinBERT:
```
python run_all.py --skip-finbert
```
To skip clustering:
```
python run_all.py --skip-clustering
```

GPU is selected automatically when CUDA-enabled PyTorch is installed. CPU also works but is slower.

## Outputs
`outputs/` contains dataset coverage analysis, TF-IDF/SBERT/FinBERT embeddings, similarity matrices, role assignments, three diagnostic-pair comparisons, clustering metrics, figures, and a Markdown research summary.
