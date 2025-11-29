import json

# Load dokumen
with open("index.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Load query
with open("query.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

relevance = {}

for qid, qtext in queries.items():
    keywords = qtext.lower().split()
    relevance[qid] = []

    for doc_id, doc_text in docs.items():
        doc_lower = doc_text.lower()
        hits = sum(1 for w in keywords if w in doc_lower)

        if hits >= 2:  # min 2 keyword cocok
            relevance[qid].append(int(doc_id.split("_")[1]))

# SIMPAN AGAR LIST TETAP HORIZONTAL
with open("relevance.json", "w", encoding="utf-8") as f:
    f.write("{\n")

    q_keys = list(relevance.keys())
    last = q_keys[-1]

    for q in q_keys:
        arr = ",".join(str(x) for x in relevance[q])  # HORIZONTAL
        if q != last:
            f.write(f'  "{q}":[{arr}],\n')
        else:
            f.write(f'  "{q}":[{arr}]\n')

    f.write("}")