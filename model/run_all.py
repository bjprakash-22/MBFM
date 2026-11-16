import argparse, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--skip-finbert",action="store_true")
p.add_argument("--skip-clustering",action="store_true")
a=p.parse_args()
def run(name, extra=[]):
    subprocess.run([sys.executable,str(R/"src"/name),*extra],cwd=R,check=True)
run("01_dataset_analysis.py")
x=["--skip-finbert"] if a.skip_finbert else []
run("02_build_embeddings.py",x)
run("03_similarity_models.py",x)
if not a.skip_clustering: run("04_clustering.py",x)
run("05_make_report.py")
print("Pipeline completed. See outputs/.")
