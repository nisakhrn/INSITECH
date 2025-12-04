import pandas as pd
import json
from collections import defaultdict

def build_inverted_index(csv_file, output_json):
    df = pd.read_csv(csv_file)

    if "isi" not in df.columns:
        raise ValueError("Kolom 'isi' tidak ditemukan di CSV!")

    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_meta = {}
    doc_length = {}

    N = len(df)

    for doc_id, row in df.iterrows():
        text = str(row["isi"])
        tokens = text.split()

        judul = row["judul"] if "judul" in df.columns else ""
        url = row["url"] if "url" in df.columns else ""
        doc_meta[str(doc_id)] = {
            "judul": judul,
            "url": url
        }

        doc_length[str(doc_id)] = len(tokens)

        for tok in tokens:
            if tok:
                inverted_index[tok][str(doc_id)] += 1

    inverted_index = {term: dict(postings)
                      for term, postings in inverted_index.items()}

    index_data = {
        "N": N,
        "doc_meta": doc_meta,
        "doc_length": doc_length,
        "inverted_index": inverted_index
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    print(f"Selesai membangun indeks untuk {N} dokumen.")
    print(f"Disimpan ke: {output_json}")

if __name__ == "__main__":
    build_inverted_index("korpus_500_preprocessed.csv", "indexing.json")