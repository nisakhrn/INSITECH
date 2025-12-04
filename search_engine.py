import json
import math

# ---------- LOAD INDEX ----------
with open("indexing.json", "r", encoding="utf-8") as f:
    INDEX = json.load(f)

INVERTED = INDEX["inverted_index"]
DOC_META = INDEX["doc_meta"]
DOC_LEN = {int(k): v for k, v in INDEX["doc_length"].items()}
N = INDEX["N"]
AVG_DL = sum(DOC_LEN.values()) / N

# ---------- PREPROCESS QUERY ----------
def preprocess_query(q: str):
    # simple: lowercase + split; bisa kamu upgrade pakai stemming kalau mau
    return q.lower().split()

# ---------- TF-IDF ----------
def compute_idf(term: str) -> float:
    if term not in INVERTED:
        return 0.0
    df = len(INVERTED[term])
    return math.log(N / (df + 1))

def search_tfidf(query: str, top_k: int = 10):
    terms = preprocess_query(query)
    scores = {}

    for term in terms:
        if term not in INVERTED:
            continue
        idf = compute_idf(term)
        postings = INVERTED[term]  # {doc_id(str): tf}

        for doc_id_str, tf in postings.items():
            doc_id = int(doc_id_str)
            scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        meta = DOC_META.get(str(doc_id), {})
        results.append({
            "doc_id": doc_id,
            "judul": meta.get("judul", ""),
            "url": meta.get("url", ""),
            "score": score
        })

    return results

# ---------- BM25 ----------
def compute_idf_bm25(term: str) -> float:
    if term not in INVERTED:
        return 0.0
    df = len(INVERTED[term])
    return math.log((N - df + 0.5) / (df + 0.5) + 1)

def search_bm25(query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
    terms = preprocess_query(query)
    scores = {}

    for term in terms:
        if term not in INVERTED:
            continue

        idf = compute_idf_bm25(term)
        postings = INVERTED[term]

        for doc_id_str, tf in postings.items():
            doc_id = int(doc_id_str)
            dl = DOC_LEN[doc_id]
            tf = float(tf)

            denom = tf + k1 * (1 - b + b * (dl / AVG_DL))
            score_inc = idf * (tf * (k1 + 1) / denom)

            scores[doc_id] = scores.get(doc_id, 0.0) + score_inc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        meta = DOC_META.get(str(doc_id), {})
        results.append({
            "doc_id": doc_id,
            "judul": meta.get("judul", ""),
            "url": meta.get("url", ""),
            "score": score
        })

    return results

if __name__ == "__main__":
    # tes cepat
    q = "kecerdasan buatan generatif"
    print("TF-IDF:", search_tfidf(q, top_k=3))
    print("BM25 :", search_bm25(q, top_k=3))