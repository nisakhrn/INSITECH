import json
from rank_bm25 import BM25Okapi

# Load dokumen
with open("index.json", "r", encoding="utf-8") as f:
    index_data = json.load(f)

documents = list(index_data.values())
doc_keys = list(index_data.keys())

tokenized = [doc.split() for doc in documents]

# Buat BM25
bm25 = BM25Okapi(tokenized)

bm25_output = {
    "avgdl": bm25.avgdl,
    "dokumen": {}
}

# Loop dokumen
for i, key in enumerate(doc_keys):
    term_freq = {}
    for token in tokenized[i]:
        term_freq[token] = term_freq.get(token, 0) + 1

    bm25_output["dokumen"][key] = {
        "length": len(tokenized[i]),
        "tf": term_freq
    }

# Simpan manual supaya list horizontal
with open("bm25.json", "w", encoding="utf-8") as f:
    f.write("{\n")
    f.write(f'  "avgdl":{bm25_output["avgdl"]},\n')
    f.write('  "dokumen":{\n')

    keys = list(bm25_output["dokumen"].keys())
    last = keys[-1]

    for k in keys:
        tf = bm25_output["dokumen"][k]["tf"]
        tf_str = ",".join([f'"{t}":{tf[t]}' for t in tf])

        if k != last:
            f.write(f'    "{k}":{{"length":{bm25_output["dokumen"][k]["length"]},"tf":{{{tf_str}}}}},\n')
        else:
            f.write(f'    "{k}":{{"length":{bm25_output["dokumen"][k]["length"]},"tf":{{{tf_str}}}}}\n')

    f.write("  }\n")
    f.write("}")