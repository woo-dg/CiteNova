# cluster_papers.py
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans

EMB_FILES = [
    "University_of_Toronto_embeddings.jsonl",
    "McMaster_University_embeddings.jsonl",
    "University_of_Waterloo_embeddings.jsonl",
    "Queens_University_embeddings.jsonl",
    "University_of_Guelph_embeddings.jsonl",
]

# Tweak cluster count (slightly tighter than sqrt(n/2))
def choose_k(n: int) -> int:
    base = math.sqrt(n)                      # start with sqrt(n)
    k = int(round(base)) + 1                 # nudge up
    return max(4, min(18, k))                # keep within a sane band

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

for emb_path in EMB_FILES:
    if not Path(emb_path).exists():
        print(f"[skip] {emb_path} not found")
        continue

    recs = list(read_jsonl(emb_path))
    ids, titles, dois, X = [], [], [], []

    for r in recs:
        v = r.get("vector")
        if isinstance(v, list):
            ids.append(r["id"])
            titles.append(r.get("title"))
            dois.append(r.get("doi"))
            X.append(v)

    if not X:
        print(f"[warn] No vectors in {emb_path}, skipping.")
        continue

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    k = choose_k(n)
    print(f"[info] {emb_path}: n={n}, k={k}")

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    out_path = emb_path.replace("_embeddings.jsonl", "_papers_with_clusters.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r, c in zip(recs, labels):
            if isinstance(r.get("vector"), list):  # keep aligned
                r2 = dict(r)
                r2["cluster_id"] = int(c)
                f.write(json.dumps(r2, ensure_ascii=False) + "\n")

    sizes = Counter(labels).most_common()
    print(f"[ok] wrote {out_path} | cluster sizes: {sizes}")
