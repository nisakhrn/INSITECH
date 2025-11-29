import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import time

# =========================
# STOPWORDS
# =========================
stop_eng = {
    "the","a","an","and","or","is","are","was","were","be","been","to","of",
    "in","on","for","from","with","that","this","it","as","at","by","but",
    "not","they","their","them","you","your","we","our","i","my","me"
}

stop_ind = {
    "yang","untuk","dengan","pada","dari","atau","dan","juga","ini",
    "itu","karena","agar","ke","di","sebagai","adalah","oleh",
    "akan","dapat","dalam","sudah","tersebut","para","merupakan","pun",
    "kami","kita","anda","saya","mereka","ia","dia"
}

STOPWORDS = stop_eng.union(stop_ind)

# =========================
# STEMMER
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =========================
# NOISE PATTERNS
# =========================
noise_patterns = [
    r"scroll to continue with content",
    r"scroll continue content",
    r"continue reading",
    r"baca juga",
    r"halaman selanjutnya",
    r"baca berita selengkapnya",
    r"lihat foto",
    r"simak selengkapnya",
]

def bersihkan_noise(text):
    text = text.lower()
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text)
    return text

# =========================
# CLEANING FUNCTIONS
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)     # remove URL
    text = re.sub(r"<.*?>", " ", text)                       # remove HTML
    text = text.lower()                                      # lowercase
    text = re.sub(r"\d+", " ", text)                         # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)                     # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                 # normalize space
    return text

def stemming(text):
    return stemmer.stem(text)

def remove_stopwords(text):
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(filtered)

# =========================
# MAIN PROCESSING
# =========================
def preprocess_and_save(input_file, output_file):
    print(" Membaca file:", input_file)
    df = pd.read_csv(input_file)

    if "isi" not in df.columns:
        raise ValueError("Kolom 'isi' tidak ditemukan!")

    total = len(df)
    print(f" Total dokumen: {total}\n")

    new_isi = []

    for i in range(total):
        raw = df.loc[i, "isi"]

        # Urutan yang benar
        text = bersihkan_noise(raw)
        text = clean_text(text)
        text = stemming(text)           # 1️⃣ Stemming dulu
        text = remove_stopwords(text)   # 2️⃣ Stopword setelah stemming

        new_isi.append(text)

        # Progress bar
        percent = (i + 1) / total * 100
        print(f" Memproses {i+1}/{total}  |  {percent:.2f}% selesai", end="\r")
        time.sleep(0.01)

    df["isi"] = new_isi

    print("\n\n Menyimpan ke:", output_file)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print("\n SELESAI — file tanpa kolom tambahan berhasil dibuat!")

# =========================
# RUN
# =========================
if __name__ == "_main_":
    preprocess_and_save("korpus_500.csv", "korpus_500_preprocessed.csv")