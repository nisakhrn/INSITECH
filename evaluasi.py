import json
import numpy as np


# ============================
#   BASIC METRIC FUNCTIONS
# ============================

def precision(relevant, retrieved):
    if len(retrieved) == 0:
        return 0.0
    return len(set(relevant) & set(retrieved)) / len(retrieved)


def recall(relevant, retrieved):
    if len(relevant) == 0:
        return 0.0
    return len(set(relevant) & set(retrieved)) / len(relevant)


def f1_score(p, r):
    if p == 0 and r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


# ============================
#   AVERAGE PRECISION (AP)
# ============================

def average_precision(relevant, ranking):
    score = 0.0
    hits = 0

    for i, doc in enumerate(ranking):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)

    if len(relevant) == 0:
        return 0.0
    
    return score / len(relevant)


# ============================
#   MAP & MRR
# ============================

def mean_average_precision(relevance, results):
    ap_scores = []
    for q in relevance:
        ap_scores.append(average_precision(relevance[q], results[q]))
    return sum(ap_scores) / len(ap_scores)


def mean_reciprocal_rank(relevance, results):
    rr_list = []

    for q in relevance:
        ranking = results[q]
        relevant = set(relevance[q])
        rr = 0
        for i, doc in enumerate(ranking):
            if doc in relevant:
                rr = 1 / (i + 1)
                break
        rr_list.append(rr)

    return sum(rr_list) / len(rr_list)


# ============================
#       nDCG@10
# ============================

def ndcg_at_k(relevant, ranking, k=10):
    dcg = 0.0
    for i, doc in enumerate(ranking[:k]):
        if doc in relevant:
            dcg += 1 / np.log2(i + 2)

    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg != 0 else 0.0


# ============================
#       LOAD FILES
# ============================

queries = json.load(open("query.json", "r", encoding="utf-8"))
relevance = json.load(open("relevance.json", "r", encoding="utf-8"))
tfidf_results = json.load(open("tfidf_result.json", "r", encoding="utf-8"))
bm25_results = json.load(open("bm25_result.json", "r", encoding="utf-8"))


# ============================
#        EVALUATE TF-IDF
# ============================

print("\n==============================")
print("🔥   EVALUASI TF-IDF")
print("==============================")

map_tfidf = mean_average_precision(relevance, tfidf_results)
mrr_tfidf = mean_reciprocal_rank(relevance, tfidf_results)

for q in queries:
    p = precision(relevance[q], tfidf_results[q])
    r = recall(relevance[q], tfidf_results[q])
    f = f1_score(p, r)
    ap = average_precision(relevance[q], tfidf_results[q])
    nd = ndcg_at_k(relevance[q], tfidf_results[q])

    print(f"\nQuery {q}:")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1 Score  : {f:.4f}")
    print(f"  AP        : {ap:.4f}")
    print(f"  nDCG@10   : {nd:.4f}")

print(f"\nMAP (TF-IDF): {map_tfidf:.4f}")
print(f"MRR (TF-IDF): {mrr_tfidf:.4f}")


# ============================
#        EVALUATE BM25
# ============================

print("\n==============================")
print("🔥   EVALUASI BM25")
print("==============================")

map_bm25 = mean_average_precision(relevance, bm25_results)
mrr_bm25 = mean_reciprocal_rank(relevance, bm25_results)

for q in queries:
    p = precision(relevance[q], bm25_results[q])
    r = recall(relevance[q], bm25_results[q])
    f = f1_score(p, r)
    ap = average_precision(relevance[q], bm25_results[q])
    nd = ndcg_at_k(relevance[q], bm25_results[q])

    print(f"\nQuery {q}:")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1 Score  : {f:.4f}")
    print(f"  AP        : {ap:.4f}")
    print(f"  nDCG@10   : {nd:.4f}")

print(f"\nMAP (BM25): {map_bm25:.4f}")
print(f"MRR (BM25): {mrr_bm25:.4f}")