import argparse, numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from common import OUT,corpus
p=argparse.ArgumentParser(); p.add_argument("--skip-finbert",action="store_true"); a=p.parse_args()
d=corpus()
raw=d.Original_Variable.astype(str).tolist()
norm=d.Normalized_Variable.astype(str).tolist()
enr=[" ".join(map(str,[r.Normalized_Variable,r.Refined_Domain,r.Refined_Category,r.Refined_Qualifier,r.Semantic_Description])) for r in d.itertuples()]
for name,txt in [("raw",raw),("normalized",norm),("ontology",enr)]:
    m=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True).fit_transform(txt); save_npz(OUT/f"tfidf_{name}.npz",m); print("TFIDF",name,m.shape)
import torch
from sentence_transformers import SentenceTransformer
device="cuda" if torch.cuda.is_available() else "cpu"
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device=device)
for name,txt in [("raw",raw),("normalized",norm),("ontology",enr)]:
    e=model.encode(txt,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True); np.save(OUT/f"sbert_{name}.npy",e); print("SBERT",name,e.shape)
if not a.skip_finbert:
    from transformers import AutoTokenizer,AutoModel
    name="ProsusAI/finbert"; tok=AutoTokenizer.from_pretrained(name); mod=AutoModel.from_pretrained(name).to(device); mod.eval()
    for key,txt in [("raw",raw),("normalized",norm),("ontology",enr)]:
        batches=[]
        with torch.no_grad():
            for i in range(0,len(txt),24):
                t=tok(txt[i:i+24],padding=True,truncation=True,max_length=96,return_tensors="pt").to(device)
                h=mod(**t).last_hidden_state; mask=t["attention_mask"].unsqueeze(-1); pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
        e=np.vstack(batches); np.save(OUT/f"finbert_{key}.npy",e); print("FinBERT",key,e.shape)
d.to_csv(OUT/"concept_corpus_used.csv",index=False)
