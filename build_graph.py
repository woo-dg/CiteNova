# build_graph.py  (a.k.a. "build_topics_and_papers")
import json, math, re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CLUSTERED_FILES = [
    "University_of_Toronto_papers_with_clusters.jsonl",
    "McMaster_University_papers_with_clusters.jsonl",
    "University_of_Waterloo_papers_with_clusters.jsonl",
    "Queens_University_papers_with_clusters.jsonl",
    "University_of_Guelph_papers_with_clusters.jsonl",
]

# mark items below this cosine-to-centroid as "weak"
WEAK_SIM_THRESHOLD = 0.34   # try 0.32–0.40 per model; tune later
MAX_TFIDF_TERMS = 6

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def uni_key_from_path(p: Path) -> str:
    # "University_of_Toronto_papers_with_clusters.jsonl" -> "University_of_Toronto"
    return re.sub(r"_papers_with_clusters\.jsonl$", "", p.name)

def cosine_centroid_and_sims(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # vecs are already normalized (from embed step). Just mean → renormalize.
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid) + 1e-12
    centroid = centroid / norm
    sims = vecs @ centroid
    return centroid, sims

def top_terms_for_cluster(titles: list[str]) -> list[str]:
    if not titles:
        return []
    # very light TF-IDF over titles to label cluster
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_features=2000,
        stop_words="english"
    )
    X = vec.fit_transform(titles)
    vocab = np.array(vec.get_feature_names_out())
    scores = np.asarray(X.sum(axis=0)).ravel()
    idx = scores.argsort()[::-1][:MAX_TFIDF_TERMS]
    terms = [vocab[i] for i in idx]
    return terms

def process_university(in_path: Path):
    uni_key = uni_key_from_path(in_path)
    recs = [r for r in read_jsonl(in_path) if isinstance(r.get("vector"), list)]
    if not recs:
        print(f"[warn] no records for {in_path}")
        return

    # group by cluster id
    clusters = defaultdict(list)
    for r in recs:
        clusters[int(r["cluster_id"])].append(r)

    topics = []
    papers = []

    for c_id, items in clusters.items():
        # vectors
        V = np.asarray([r["vector"] for r in items], dtype=np.float32)
        # compute centroid + sims
        centroid, sims = cosine_centroid_and_sims(V)

        # split strong/weak
        strong_items, weak_items = [], []
        for r, sim in zip(items, sims):
            simf = float(sim)
            bucket = strong_items if simf >= WEAK_SIM_THRESHOLD else weak_items
            bucket.append((r, simf))

        titles_for_terms = [r.get("title") or "" for r, _ in strong_items] or [r.get("title") or "" for r, _ in weak_items]
        top_terms = top_terms_for_cluster(titles_for_terms)
        label = ", ".join(top_terms[:3]) if top_terms else f"Cluster {c_id}"

        # topics bubble
        topics.append({
            "cluster_id": int(c_id),
            "label": label,
            "size": len(items),
            "weak_count": len(weak_items),
        })

        # papers list (strong first)
        for bucket, quality in [(strong_items, "strong"), (weak_items, "weak")]:
            for r, simf in sorted(bucket, key=lambda x: -x[1]):
                papers.append({
                    "id": r["id"],
                    "title": r.get("title"),
                    "doi": r.get("doi"),
                    "cluster_id": int(c_id),
                    "sim_to_centroid": simf,
                    "quality": quality
                })

    # Sort UX
    topics.sort(key=lambda t: t["size"], reverse=True)
    papers.sort(key=lambda p: (p["cluster_id"], -p["sim_to_centroid"], (p["title"] or "")))

    out_topics = f"{uni_key}_topics.json"
    out_papers = f"{uni_key}_papers.json"
    with open(out_topics, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    with open(out_papers, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"[ok] → {out_topics} ({len(topics)} bubbles), {out_papers} ({len(papers)} papers)")

if __name__ == "__main__":
    for p in CLUSTERED_FILES:
        path = Path(p)
        if path.exists():
            process_university(path)
        else:
            print(f"[skip] {p} not found")
