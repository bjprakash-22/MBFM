
# Accounting-Aware Semantic Alignment of Financial Statement Concepts Using Transformer Embeddings and Ontology-Guided Similarity

Banavathu Jothi Prakash\
banavathu23@iiserb.ac.in\
Indian Institute of Science Education and Research Bhopal, India

## Project Structure

```text
model/
  data/
  src/
  outputs/
  run_all.py
```

## Windows Setup

Open PowerShell in the project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run Everything

```bash
python run_all.py
```

The first run downloads `all-MiniLM-L6-v2` and `ProsusAI/finbert` from Hugging Face.

To skip FinBERT:

```bash
python run_all.py --skip-finbert
```

To skip clustering:

```bash
python run_all.py --skip-clustering
```

GPU is selected automatically when CUDA-enabled PyTorch is installed. CPU also works but is slower.

## Outputs

`outputs/` contains dataset coverage analysis, TF-IDF/SBERT/FinBERT embeddings, similarity matrices, role assignments, diagnostic-pair comparisons, clustering metrics, figures, and a Markdown research summary.
