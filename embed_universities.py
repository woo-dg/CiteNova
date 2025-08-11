# embed_universities.py
import os, json
from pathlib import Path
from typing import List, Dict, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Input files produced earlier (id, title, doi, abstract)
INPUT_FILES = [
    "University_of_Toronto.json",
    "McMaster_University.json",
    "University_of_Waterloo.json",
    "Queens_University.json",
    "University_of_Guelph.json",
]

# ==== MODEL CHOICE (free, 512 tokens) ====
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # alt: "intfloat/e5-small-v2"
MAX_TOKENS = 512
USE_E5_PREFIX = False                   # set True only if you switch to e5

# ---------------- utils ----------------
def load_items(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: str, rows: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def compose_text(tokenizer, title: str | None, abstract: str | None, max_tokens: int = MAX_TOKENS) -> str:
    base = (title or "").strip()
    if abstract and abstract.strip():
        base = f"{base}. {abstract.strip()}" if base else abstract.strip()
    if USE_E5_PREFIX:
        base = f"passage: {base}"
    # truncate via tokenizer so special tokens are counted correctly
    ids = tokenizer.encode(base, add_special_tokens=True, truncation=True, max_length=max_tokens)
    return tokenizer.decode(ids, skip_special_tokens=True)

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    # normalize=True → cosine-friendly
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float32)

# ---------------- main ----------------
if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    for infile in INPUT_FILES:
        if not Path(infile).exists():
            print(f"[skip] {infile} not found")
            continue

        items = load_items(infile)
        ids, titles, dois, texts = [], [], [], []

        for r in items:
            pid = r.get("id")
            title = r.get("title") or ""
            doi = r.get("doi")
            abstract = r.get("abstract")
            text = compose_text(tokenizer, title, abstract, MAX_TOKENS)
            if not text.strip():
                continue

            ids.append(pid)
            titles.append(title)
            dois.append(doi)
            texts.append(text)

        if not texts:
            print(f"[warn] no texts for {infile}")
            continue

        vecs = embed_texts(model, texts, batch_size=64)

        out_rows = []
        for pid, title, doi, vec in zip(ids, titles, dois, vecs):
            out_rows.append({
                "id": pid,
                "title": title,
                "doi": doi,
                "vector": vec.tolist()
            })

        out_path = infile.replace(".json", "_embeddings.jsonl")
        write_jsonl(out_path, out_rows)
        print(f"[ok] Wrote {len(out_rows)} embeddings → {out_path}")
