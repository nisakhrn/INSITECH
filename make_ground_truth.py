import json
from search_engine import search_bm25  # bisa diganti search_tfidf

GROUND_FILE = "ground_truth.json"
TOP_K = 10

TEST_QUERIES = [
    "teknologi digital",
    "data cloud",
    "internet"
]

def load_ground_truth():
    try:
        with open(GROUND_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_ground_truth(gt):
    with open(GROUND_FILE, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print(f"\nGround truth tersimpan ke {GROUND_FILE}")

def annotate_query(query, gt):
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    results = search_bm25(query, top_k=TOP_K)

    if not results:
        print("Tidak ada dokumen yang ditemukan untuk query ini.")
        gt[query] = []
        return

    for idx, r in enumerate(results, start=1):
        print(f"[{idx}] doc_id={r['doc_id']}")
        print(f"     Judul : {r.get('judul', '')}")
        print(f"     URL   : {r.get('url', '')}")
        print("-" * 60)

    print("\nMasukkan NOMOR dokumen yang RELEVAN untuk query ini,")
    print("pisahkan dengan koma, misal: 1,3,5  (kosongkan jika tidak ada).")

    inp = input("Nomor relevan: ").strip()
    if not inp:
        rel_doc_ids = []
    else:
        indices = []
        for x in inp.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                i = int(x)
                if 1 <= i <= len(results):
                    indices.append(i)
            except ValueError:
                continue
        rel_doc_ids = [results[i-1]["doc_id"] for i in indices]

    gt[query] = rel_doc_ids
    print(f"Relevan untuk '{query}': {rel_doc_ids}")

def main():
    gt = load_ground_truth()
    print("=== Pembuat Ground Truth (5 Query Tetap) ===")
    print("Ground truth awal memuat", len(gt), "query.\n")

    for q in TEST_QUERIES:
        annotate_query(q, gt)
        save_ground_truth(gt)

    print("\nSelesai. Total query di ground_truth.json:", len(gt))

if __name__ == "__main__":
    main()