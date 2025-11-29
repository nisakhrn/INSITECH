import pandas as pd

df = pd.read_csv("korpus_500_preprocessed.csv")
documents = df["isi"].astype(str).tolist()

inverted = {}

# Bangun inverted index
for doc_id, text in enumerate(documents):
    for term in text.split():
        if term not in inverted:
            inverted[term] = []
        if doc_id not in inverted[term]:
            inverted[term].append(doc_id)

# Simpan manual supaya format seperti yang kamu mau
with open("inverted_index.json", "w", encoding="utf-8") as f:
    f.write("{\n")  # buka JSON

    terms = list(inverted.keys())
    last_term = terms[-1]

    for term in terms:
        doc_list = ",".join(str(x) for x in inverted[term])  # horizontal list

        if term != last_term:
            f.write(f'  "{term}":[{doc_list}],\n')
        else:
            f.write(f'  "{term}":[{doc_list}]\n')

    f.write("}")