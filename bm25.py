import json
from rank_bm25 import BM25Okapi
import numpy as np

# Load dokumen
with open("index.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

doc_list = list(docs.values())
tokenized = [d.split() for d in doc_list]

bm25 = BM25Okapi(tokenized)

# Load query
with open("query.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

results = {}

for q_name, q_text in queries.items():
    q_tokens = q_text.split()
    scores = bm25.get_scores(q_tokens)

    ranking = [int(x) for x in np.argsort(scores)[::-1]]
    results[q_name] = ranking

# 🔥 SIMPAN HORIZONTAL
with open("bm25_result.json", "w", encoding="utf-8") as f:
    f.write("{\n")

    keys = list(results.keys())
    last = keys[-1]

    for k in keys:
        arr_str = ",".join(str(x) for x in results[k])
        if k != last:
            f.write(f'  "{k}":[{arr_str}],\n')
        else:
            f.write(f'  "{k}":[{arr_str}]\n')

    f.write("}")