import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="INSITECH - Mesin Pencari Dokumen",
    page_icon="🔍",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 50%, #dbeafe 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title Styling */
    .main-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        margin-top: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #075985;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 600;
        line-height: 1.6;
    }
    
    /* Search Box Container */
    .search-container {
        max-width: 1000px;
        margin: 0 auto 3rem auto;
    }
    
    /* Input Label */
    .stTextInput > label {
        color: #0c4a6e !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: white !important;
        border: 3px solid #93c5fd !important;
        border-radius: 50px !important;
        padding: 1.5rem 2rem !important;
        font-size: 1.2rem !important;
        color: #0c4a6e !important;
        font-weight: 500 !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #7dd3fc !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.35) !important;
        border-color: #3b82f6 !important;
        outline: none !important;
    }
    
    /* SelectBox Label */
    .stSelectbox > label {
        color: #0c4a6e !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* SelectBox Styling */
    .stSelectbox > div > div {
        background: white !important;
        border: 2px solid #93c5fd !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15) !important;
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }
    
    /* Slider Label */
    .stSlider > label {
        color: #0c4a6e !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Slider Styling */
    .stSlider [data-baseweb="slider"] {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #3b82f6 !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1.7rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.5) !important;
    }
    
    /* Results Card */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
        border: 2px solid #bfdbfe;
    }
    
    .result-card:hover {
        box-shadow: 0 10px 35px rgba(59, 130, 246, 0.25);
        transform: translateY(-3px);
        border-color: #60a5fa;
    }
    
    .result-rank {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        color: white;
        font-weight: 800;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-size: 1rem;
        margin-right: 1rem;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.3);
    }
    
    .result-title {
        color: #1e3a8a;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0.8rem 0;
        line-height: 1.5;
    }
    
    .result-score {
        color: #0891b2;
        font-weight: 700;
        font-size: 1rem;
        background: #cffafe;
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 15px;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e3a8a;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
        border: 2px solid #bfdbfe;
    }
    
    /* Info Box */
    .info-box {
        background: white;
        border-left: 6px solid #0891b2;
        border-radius: 15px;
        padding: 1.5rem 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
        color: #0c4a6e;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    .info-box strong {
        color: #1e3a8a;
        font-size: 1.1rem;
    }
    
    /* Comparison Column */
    .comparison-column {
        background: white;
        border-radius: 25px;
        padding: 2rem;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
        border: 2px solid #bfdbfe;
        min-height: 400px;
    }
    
    .comparison-column h3 {
        color: #1e3a8a !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 3px solid #93c5fd;
    }
    
    /* Warning Messages */
    .stWarning {
        background: white !important;
        border-left: 6px solid #f59e0b !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        color: #92400e !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2) !important;
    }
    
    /* Footer Styling */
    .footer-text {
        text-align: center;
        color: #0c4a6e;
        padding: 2rem;
        font-size: 1rem;
        font-weight: 600;
        background: white;
        border-radius: 15px;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================
@st.cache_resource
def load_models():
    with open("index.json", "r", encoding="utf-8") as f:
        index_data = json.load(f)

    documents = list(index_data.values())
    doc_keys = list(index_data.keys())

    # Preprocessing
    tokenized_docs = [doc.split() for doc in documents]

    # TF-IDF Model
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # BM25 Model
    bm25 = BM25Okapi(tokenized_docs)
    
    return tfidf_vectorizer, tfidf_matrix, bm25, doc_keys

tfidf_vectorizer, tfidf_matrix, bm25, doc_keys = load_models()

# ==========================
# FUNGSI RETRIEVAL
# ==========================
def search_tfidf(query, topk=10):
    q_vec = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().flatten()
    ranked = np.argsort(scores)[::-1][:topk]
    return [(doc_keys[i], float(scores[i])) for i in ranked]

def search_bm25(query, topk=10):
    tokenized_q = query.split()
    scores = bm25.get_scores(tokenized_q)
    ranked = np.argsort(scores)[::-1][:topk]
    return [(doc_keys[i], float(scores[i])) for i in ranked]

# ==========================
# HEADER
# ==========================
st.markdown('<h1 class="main-title">INSITECH</h1>', unsafe_allow_html=True)
st.markdown('''<p class="subtitle">
MESIN PENCARI DOKUMEN TEKNOLOGI DAN AI<br/>
MENGGUNAKAN ALGORITMA TF-IDF DAN BM25
</p>''', unsafe_allow_html=True)

# ==========================
# SEARCH INTERFACE
# ==========================
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Query Input
query = st.text_input("🔎 Masukkan Query Pencarian:", placeholder="Contoh: artificial intelligence, machine learning, deep learning...")

# Controls in 3 columns
col_method, col_topk, col_button = st.columns([2, 1, 1])

with col_method:
    method = st.selectbox(
        "Pilih Algoritma Pencarian:",
        ["TF-IDF", "BM25", "Bandingkan Keduanya"],
        index=0
    )

with col_topk:
    topk = st.slider("Jumlah Hasil (Top-K):", 5, 50, 10)

with col_button:
    search_button = st.button("Cari Sekarang")

st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# SEARCH RESULTS
# ==========================
if search_button:
    if query.strip() == "":
        st.warning("⚠️ Query tidak boleh kosong. Silakan masukkan kata kunci pencarian.")
    else:
        st.markdown("---")
        
        if method == "TF-IDF":
            st.markdown('<h2 class="section-header">Hasil Pencarian TF-IDF</h2>', unsafe_allow_html=True)
            results = search_tfidf(query, topk)
            
            if results:
                for rank, (doc, score) in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-rank">#{rank}</span>
                        <div class="result-title">{doc}</div>
                        <div class="result-score">📈 Skor: {score:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Tidak ada hasil yang ditemukan.")

        elif method == "BM25":
            st.markdown('<h2 class="section-header">Hasil Pencarian BM25</h2>', unsafe_allow_html=True)
            results = search_bm25(query, topk)
            
            if results:
                for rank, (doc, score) in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-rank">#{rank}</span>
                        <div class="result-title">{doc}</div>
                        <div class="result-score">📈 Skor: {score:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Tidak ada hasil yang ditemukan.")

        else:
            st.markdown('<h2 class="section-header">Perbandingan TF-IDF vs BM25</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")

            # TF-IDF Column
            with col1:
                st.markdown('<div class="comparison-column">', unsafe_allow_html=True)
                st.markdown("### TF-IDF")
                r1 = search_tfidf(query, topk)
                if r1:
                    for rank, (doc, score) in enumerate(r1, 1):
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="result-rank">#{rank}</span>
                            <div class="result-title">{doc}</div>
                            <div class="result-score">Skor: {score:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Tidak ada hasil.")
                st.markdown('</div>', unsafe_allow_html=True)

            # BM25 Column
            with col2:
                st.markdown('<div class="comparison-column">', unsafe_allow_html=True)
                st.markdown("### BM25")
                r2 = search_bm25(query, topk)
                if r2:
                    for rank, (doc, score) in enumerate(r2, 1):
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="result-rank">#{rank}</span>
                            <div class="result-title">{doc}</div>
                            <div class="result-score">Skor: {score:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Tidak ada hasil.")
                st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# FOOTER INFO
# ==========================
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>💡 Tentang Sistem INSITECH:</strong><br/><br/>
    INSITECH menggunakan dua algoritma pencarian terkemuka untuk penelusuran dokumen:<br/><br/>
    • <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong>: Algoritma perangkingan berbasis frekuensi kemunculan term dalam dokumen<br/>
    • <strong>BM25 (Best Matching 25)</strong>: Algoritma perangkingan probabilistik yang lebih canggih dengan mempertimbangkan panjang dokumen<br/><br/>
    Sistem ini dirancang khusus untuk penelusuran dokumen teknologi dan kecerdasan buatan (AI).
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    © 2025 INSITECH - Sistem Penelusuran Informasi Dokumen Teknologi dan AI
</div>
""", unsafe_allow_html=True)