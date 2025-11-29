import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load index.json
with open("index.json", "r", encoding="utf-8") as f:
    index_data = json.load(f)

documents = list(index_data.values())
doc_keys = list(index_data.keys())

# Buat TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
tfidf_matrix = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()

tfidf_output = {}

# Loop setiap dokumen
for i, key in enumerate(doc_keys):
    row = tfidf_matrix[i].toarray().flatten()
    nonzero = {features[j]: float(row[j]) for j in row.nonzero()[0]}
    tfidf_output[key] = nonzero

# Simpan manual agar format rapi
with open("tfidf.json", "w", encoding="utf-8") as f:
    f.write("{\n")
    keys = list(tfidf_output.keys())
    last_key = keys[-1]

    for k in keys:
        inner = tfidf_output[k]
        inner_str = ",".join([f'"{w}":{inner[w]}' for w in inner])

        if k != last_key:
            f.write(f'  "{k}":{{{inner_str}}},\n')
        else:
            f.write(f'  "{k}":{{{inner_str}}}\n')

    f.write("}")