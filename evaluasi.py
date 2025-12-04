# eval_ir.py
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# SESUAIKAN IMPORT BERIKUT DENGAN LOKASI FUNGSI SEARCH-MU
from search_engine import search_tfidf, search_bm25
# atau kalau ada di app.py:
# from app import search_tfidf, search_bm25

TOP_K = 10
GROUND_FILE = "ground_truth.json"

# ----- LOAD GROUND TRUTH -----
with open(GROUND_FILE, "r", encoding="utf-8") as f:
    GROUND_TRUTH = json.load(f)  # {query: [relevant_doc_ids]}


def eval_query(results, relevant_ids, k=TOP_K):
    """Hitung Precision@k, Recall@k, F1, dan Average Precision untuk 1 query."""
    retrieved = [r["doc_id"] for r in results[:k]]
    rel_set = set(relevant_ids)

    true_pos = sum(1 for d in retrieved if d in rel_set)
    prec = true_pos / k if k > 0 else 0.0
    rec = true_pos / len(rel_set) if len(rel_set) > 0 else 0.0

    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    # Average Precision
    hit = 0
    ap_sum = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in rel_set:
            hit += 1
            ap_sum += hit / i
    ap = ap_sum / len(rel_set) if len(rel_set) > 0 else 0.0

    return prec, rec, f1, ap


def evaluate_model(search_func, model_name):
    """
    Evaluasi 1 model untuk semua query di ground_truth.json.
    Mengembalikan:
      - metrics_per_q: list (query, P, R, F1, AP)
      - y_true_all, y_pred_all: untuk confusion matrix
    """
    metrics_per_q = []
    y_true_all = []
    y_pred_all = []

    for query, rel_ids in GROUND_TRUTH.items():
        results = search_func(query, top_k=TOP_K)
        prec, rec, f1, ap = eval_query(results, rel_ids, k=TOP_K)
        metrics_per_q.append((query, prec, rec, f1, ap))

        # data untuk confusion matrix: semua dokumen di top_k dianggap prediksi "relevan"
        rel_set = set(rel_ids)
        retrieved = [r["doc_id"] for r in results[:TOP_K]]

        for d in retrieved:
            y_true_all.append(1 if d in rel_set else 0)  # 1 = relevan, 0 = tidak
            y_pred_all.append(1)                         # semua di top_k = diprediksi relevan

    return metrics_per_q, np.array(y_true_all), np.array(y_pred_all)


def plot_bar(metrics_tfidf, metrics_bm25):
    """
    Buat grafik bar Precision, Recall, F1
    dengan pasangan (TF-IDF, BM25) untuk tiap query.
    """
    labels = []
    P_vals = []
    R_vals = []
    F_vals = []

    # Asumsikan urutan query di kedua list sama
    for (q1, P_t, R_t, F1_t, _), (q2, P_b, R_b, F1_b, _) in zip(metrics_tfidf, metrics_bm25):
        query = q1  # nama query

        labels.append(f"TF-IDF {query}")
        P_vals.append(P_t); R_vals.append(R_t); F_vals.append(F1_t)

        labels.append(f"BM25 {query}")
        P_vals.append(P_b); R_vals.append(R_b); F_vals.append(F1_b)

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, P_vals, width, label="Precision", color="#4C72B0")  # biru
    plt.bar(x,         R_vals, width, label="Recall",    color="#55A868")  # hijau
    plt.bar(x + width, F_vals, width, label="F1-Score",  color="#C44E52")  # merah


    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Visualisasi Perbandingan IR Model Berdasarkan Kueri")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("perbandingan_metric_per_query.png", dpi=300)
    plt.close()


def plot_confusion(y_true, y_pred, model_name):
    """Plot confusion matrix sederhana untuk 1 model."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Relevan", "Pred Tidak"])
    ax.set_yticklabels(["Aktual Relevan", "Aktual Tidak"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"confusion_{model_name}.png", dpi=300)
    plt.close()


def plot_map(map_tfidf, map_bm25):
    """Plot bar chart khusus MAP kedua model."""
    models = ["TF-IDF", "BM25"]
    values = [map_tfidf, map_bm25]

    x = np.arange(len(models))
    width = 0.4

    plt.figure(figsize=(5, 4))
    plt.bar(x, values, width, color=["skyblue", "orange"])
    plt.xticks(x, models)
    plt.ylabel("MAP")
    plt.ylim(0, 1.05)
    plt.title("Perbandingan MAP TF-IDF vs BM25")
    plt.tight_layout()
    plt.savefig("map_tfidf_vs_bm25.png", dpi=300)
    plt.close()


def main():
    # Evaluasi kedua model
    m_tfidf, y_true_tfidf, y_pred_tfidf = evaluate_model(search_tfidf, "TF-IDF")
    m_bm25,  y_true_bm25,  y_pred_bm25  = evaluate_model(search_bm25,  "BM25")

    # Hitung MAP
    map_tfidf = np.mean([x[4] for x in m_tfidf])
    map_bm25  = np.mean([x[4] for x in m_bm25])

    print("=== Hasil Per Query (TF-IDF) ===")
    print("Query\t\tPrecision\tRecall\t\tF1-Score\tAP")
    for q, P, R, F1, AP in m_tfidf:
        print(f"{q}\t{P:.4f}\t\t{R:.4f}\t\t{F1:.4f}\t\t{AP:.4f}")

    print("\n=== Hasil Per Query (BM25) ===")
    print("Query\t\tPrecision\tRecall\t\tF1-Score\tAP")
    for q, P, R, F1, AP in m_bm25:
        print(f"{q}\t{P:.4f}\t\t{R:.4f}\t\t{F1:.4f}\t\t{AP:.4f}")

    print("\nMAP TF-IDF:", round(map_tfidf, 4))
    print("MAP BM25 :", round(map_bm25, 4))

    # Grafik
    plot_bar(m_tfidf, m_bm25)
    plot_confusion(y_true_tfidf, y_pred_tfidf, "TF-IDF")
    plot_confusion(y_true_bm25,  y_pred_bm25,  "BM25")
    plot_map(map_tfidf, map_bm25)

if __name__ == "__main__":
    main()