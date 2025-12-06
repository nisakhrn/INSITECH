import streamlit as st
import json
import numpy as np
import pandas as pd
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
        background: #052659;
        color: #FFFFFF;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Animated Tech Background */
    .tech-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        opacity: 0.5;
    }
    
    /* Moving particles */
    .particle {
        position: absolute;
        width: 8px;
        height: 8px;
        background: #89CFF0;
        border-radius: 50%;
        animation: float 20s infinite linear;
        box-shadow: 0 0 20px rgba(137, 207, 240, 1), 0 0 40px rgba(137, 207, 240, 0.5);
    }
    
    .particle:nth-child(1) { left: 10%; animation-duration: 15s; animation-delay: 0s; }
    .particle:nth-child(2) { left: 20%; animation-duration: 18s; animation-delay: 2s; }
    .particle:nth-child(3) { left: 30%; animation-duration: 22s; animation-delay: 4s; }
    .particle:nth-child(4) { left: 40%; animation-duration: 16s; animation-delay: 1s; }
    .particle:nth-child(5) { left: 50%; animation-duration: 20s; animation-delay: 3s; }
    .particle:nth-child(6) { left: 60%; animation-duration: 19s; animation-delay: 5s; }
    .particle:nth-child(7) { left: 70%; animation-duration: 17s; animation-delay: 2s; }
    .particle:nth-child(8) { left: 80%; animation-duration: 21s; animation-delay: 4s; }
    .particle:nth-child(9) { left: 90%; animation-duration: 23s; animation-delay: 1s; }
    
    @keyframes float {
        0% {
            transform: translateY(100vh) translateX(0);
            opacity: 0;
        }
        10% {
            opacity: 0.8;
        }
        90% {
            opacity: 0.8;
        }
        100% {
            transform: translateY(-100px) translateX(100px);
            opacity: 0;
        }
    }
    
    /* Grid lines */
    .grid-lines {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(137, 207, 240, 0.12) 2px, transparent 2px),
            linear-gradient(90deg, rgba(137, 207, 240, 0.12) 2px, transparent 2px);
        background-size: 60px 60px;
        animation: gridMove 20s linear infinite;
    }
    
    @keyframes gridMove {
        0% {
            transform: translateY(0);
        }
        100% {
            transform: translateY(60px);
        }
    }
    
    /* Diagonal tech lines */
    .tech-line {
        position: absolute;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(137, 207, 240, 0.8), transparent);
        animation: moveLine 8s linear infinite;
        box-shadow: 0 0 10px rgba(137, 207, 240, 0.6);
    }
    
    .tech-line:nth-child(1) { 
        top: 20%; 
        width: 500px; 
        left: -500px;
        animation-delay: 0s;
    }
    .tech-line:nth-child(2) { 
        top: 50%; 
        width: 600px; 
        left: -600px;
        animation-delay: 3s;
    }
    .tech-line:nth-child(3) { 
        top: 80%; 
        width: 550px; 
        left: -550px;
        animation-delay: 6s;
    }
    
    @keyframes moveLine {
        0% {
            left: -600px;
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            left: 100%;
            opacity: 0;
        }
    }
    
    /* Ensure content is above background */
    .stApp > div {
        position: relative;
        z-index: 1;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Container */
    .header-container {
        text-align: center;
        padding: 0 2rem 1.5rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
    }
    
    /* Glowing effect background */
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(137, 207, 240, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: translateX(-50%) scale(1); }
        50% { opacity: 0.8; transform: translateX(-50%) scale(1.1); }
    }
    
    /* Algorithm Selector in Top Right */
    .algorithm-selector {
        position: fixed;
        top: 0.5rem;
        right: 0.5rem;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 8px;
        border: 2px solid rgba(137, 207, 240, 0.6);
        box-shadow: 0 4px 6px rgba(0, 103, 165, 0.3);
    }
    
    /* Title Styling - Tech Style */
    .main-title {
        text-align: center;
        font-size: 6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #89CFF0 0%, #FFFFFF 50%, #89CFF0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
        letter-spacing: 0.1em;
        text-shadow: 0 0 40px rgba(137, 207, 240, 0.8);
        position: relative;
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    
    .subtitle {
        text-align: center;
        color: #FFFFFF;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        font-weight: 500;
        line-height: 1.6;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Search Box Container */
    .search-container {
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
        padding: 0;
    }
    
    /* Input Label */
    .stTextInput > label {
        display: none !important;
    }
    
    /* Remove ALL white containers and wrappers */
    .stTextInput {
        background: transparent !important;
        overflow: visible !important;
        display: flex !important;
        justify-content: center !important;
    }
    
    .stTextInput > div {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        overflow: visible !important;
        display: flex !important;
        justify-content: center !important;
    }
    
    .stTextInput > div > div {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        overflow: visible !important;
        border-radius: 30px !important;
        max-width: 600px !important;
        width: 100% !important;
    }
    
    /* Input Styling - Tech Futuristic */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(137, 207, 240, 0.6) !important;
        border-radius: 30px !important;
        padding: 0.9rem 1.75rem !important;
        font-size: 1rem !important;
        color: #0067A5 !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(123, 187, 255, 0.2) !important;
        transition: all 0.3s ease !important;
        line-height: 1.5 !important;
        min-height: 48px !important;
        height: 48px !important;
        width: 100% !important;
        display: block !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #4997D0 !important;
        font-weight: 400 !important;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: rgba(73, 151, 208, 0.8) !important;
        box-shadow: 0 8px 24px rgba(137, 207, 240, 0.5) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #89CFF0 !important;
        box-shadow: 0 0 0 3px rgba(137, 207, 240, 0.3), 0 8px 32px rgba(73, 151, 208, 0.6) !important;
        outline: none !important;
    }
    
    /* SelectBox Label */
    .stSelectbox > label {
        font-size: 0.75rem !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* SelectBox Styling - Tech Theme */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(137, 207, 240, 0.5) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(123, 187, 255, 0.2) !important;
        color: #6B63B5 !important;
        font-weight: 400 !important;
        font-size: 0.875rem !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #6B63B5 !important;
        font-weight: 400 !important;
        font-size: 0.875rem !important;
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
        background: linear-gradient(135deg, #0067A5 0%, #4997D0 100%) !important;
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
    
    /* Results Card - Tech Style */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 103, 165, 0.3);
        transition: all 0.3s ease;
        border: 2px solid rgba(137, 207, 240, 0.4);
    }
    
    .result-card:hover {
        box-shadow: 0 8px 24px rgba(73, 151, 208, 0.5);
        transform: translateY(-2px);
        border-color: rgba(137, 207, 240, 0.7);
    }
    
    .result-rank {
        display: inline-block;
        background: linear-gradient(135deg, #0067A5 0%, #4997D0 100%);
        color: white;
        font-weight: 700;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0, 103, 165, 0.4);
    }
    
    .result-url {
        color: #4997D0;
        font-size: 0.875rem;
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }
    
    .result-source {
        background: rgba(137, 207, 240, 0.25);
        color: #0067A5;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        display: inline-block;
        margin-right: 0.5rem;
        text-transform: uppercase;
        border: 1px solid rgba(123, 187, 255, 0.4);
    }
    
    .result-title {
        color: #0067A5;
        font-weight: 600;
        font-size: 1.25rem;
        margin: 0.5rem 0;
        line-height: 1.4;
        cursor: pointer;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    .result-title:hover {
        color: #4997D0;
        text-decoration: underline;
    }
    
    .result-snippet {
        color: #367588;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0.6rem 0;
    }
    
    .result-score {
        color: #0067A5;
        font-weight: 600;
        font-size: 0.85rem;
        background: rgba(137, 207, 240, 0.25);
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        margin-top: 0.5rem;
        border: 1px solid rgba(73, 151, 208, 0.4);
    }
    
    /* Section Headers - Tech Style */
    .section-header {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
        margin-top: 2rem;
        padding-left: 0.75rem;
        border-left: 4px solid #89CFF0;
        text-shadow: 0 0 20px rgba(137, 207, 240, 0.5);
    }
    
    /* Info Box - Tech Style */
    .info-box {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-left: 4px solid #7BBBFF;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 2rem auto;
        max-width: 900px;
        color: #6B63B5;
        font-size: 0.875rem;
        line-height: 1.6;
        border: 2px solid rgba(123, 187, 255, 0.3);
        box-shadow: 0 4px 12px rgba(123, 187, 255, 0.3);
    }
    
    .info-box strong {
        color: #8B7FFF;
        font-size: 0.9rem;
    }
    
    /* Comparison Column */
    .comparison-column {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        min-height: 300px;
    }
    
    .comparison-column h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-align: center;
        margin-bottom: 1rem !important;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
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
    
    /* Footer Styling - Tech Theme */
    .footer-text {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        color: #FFFFFF;
        padding: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        background: rgba(5, 38, 89, 0.95);
        backdrop-filter: blur(10px);
        border-top: 2px solid rgba(137, 207, 240, 0.4);
        z-index: 1000;
        text-shadow: 0 0 10px rgba(123, 187, 255, 0.2);
    }
    
    /* Pagination Styling - Tech Theme */
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.3rem;
        margin: 2rem auto;
        padding: 0.8rem;
        max-width: 600px;
    }
    
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        color: #0067A5 !important;
        border: 2px solid rgba(137, 207, 240, 0.5) !important;
        border-radius: 4px !important;
        padding: 0.4rem 0.8rem !important;
        min-width: 36px !important;
        height: 36px !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        margin: 0 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(137, 207, 240, 0.3) !important;
        border-color: #89CFF0 !important;
        box-shadow: 0 0 12px rgba(137, 207, 240, 0.6) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0067A5 0%, #4997D0 100%) !important;
        color: white !important;
        border: 2px solid rgba(137, 207, 240, 0.5) !important;
        border-radius: 4px !important;
        padding: 0.4rem 0.8rem !important;
        min-width: 36px !important;
        height: 36px !important;
        font-size: 0.875rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(0, 103, 165, 0.6) !important;
        margin: 0 !important;
    }
    
    .page-info {
        color: #FFFFFF;
        font-size: 0.8rem;
        margin: 0.75rem 0 1rem 0;
        text-align: left;
        padding-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    """Load korpus CSV untuk pencarian dan detail dokumen"""
    df = pd.read_csv("korpus_500_preprocessed.csv")
    
    # Buat mapping untuk detail dokumen
    doc_details = {}
    documents = []
    doc_keys = []
    
    for idx, row in df.iterrows():
        doc_name = f"dokumen_{idx}"
        doc_keys.append(doc_name)
        
        # Simpan dokumen yang sudah dipreprocess untuk pencarian
        documents.append(row['isi'])
        
        # Simpan detail lengkap untuk tampilan hasil
        doc_details[doc_name] = {
            'judul': row['judul'],
            'url': row['url'],
            'isi': row['isi'],
            'sumber': row['sumber']
        }
    
    # Preprocessing untuk tokenization
    tokenized_docs = [doc.split() for doc in documents]

    # TF-IDF Model
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # BM25 Model
    bm25 = BM25Okapi(tokenized_docs)
    
    return tfidf_vectorizer, tfidf_matrix, bm25, doc_keys, doc_details

tfidf_vectorizer, tfidf_matrix, bm25, doc_keys, corpus_details = load_data()

# ==========================
# FUNGSI RETRIEVAL
# ==========================
def get_snippet(text, query, max_length=200):
    """Ekstrak snippet yang relevan dari text berdasarkan query"""
    text = str(text)
    query_words = query.lower().split()
    text_lower = text.lower()
    
    # Cari posisi pertama kata query muncul
    best_pos = 0
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = pos
            break
    
    # Ambil snippet di sekitar posisi tersebut
    start = max(0, best_pos - 50)
    end = min(len(text), start + max_length)
    
    snippet = text[start:end]
    
    # Tambahkan ellipsis jika dipotong
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet.strip()

def search_tfidf(query, topk=None):
    q_vec = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().flatten()
    # Urutkan semua hasil
    ranked = np.argsort(scores)[::-1]
    # Filter hasil dengan score > 0 (relevan)
    results = [(doc_keys[i], float(scores[i])) for i in ranked if scores[i] > 0]
    # Batasi jika topk ditentukan
    if topk:
        results = results[:topk]
    return results

def search_bm25(query, topk=None):
    tokenized_q = query.split()
    scores = bm25.get_scores(tokenized_q)
    # Urutkan semua hasil
    ranked = np.argsort(scores)[::-1]
    # Filter hasil dengan score > 0 (relevan)
    results = [(doc_keys[i], float(scores[i])) for i in ranked if scores[i] > 0]
    # Batasi jika topk ditentukan
    if topk:
        results = results[:topk]
    return results

def display_result(doc_name, score, rank, query):
    """Display hasil pencarian dengan format mirip Google"""
    if doc_name in corpus_details:
        details = corpus_details[doc_name]
        judul = details['judul']
        url = details['url']
        isi = details['isi']
        sumber = details['sumber']
        
        # Dapatkan snippet yang relevan
        snippet = get_snippet(isi, query, max_length=180)
        
        # Format URL untuk ditampilkan (domain only)
        display_url = url.replace('https://', '').replace('http://', '')
        if len(display_url) > 60:
            display_url = display_url[:60] + "..."
        
        st.markdown(f"""
        <div class="result-card">
            <span class="result-rank">#{rank}</span>
            <div class="result-url">
                <span class="result-source">{sumber}</span>
                <span style="color: #5f6368;">{display_url}</span>
            </div>
            <a href="{url}" target="_blank" class="result-title">{judul}</a>
            <div class="result-snippet">{snippet}</div>
            <span class="result-score">📊 Skor Relevansi: {score:.4f}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback jika detail tidak ditemukan
        st.markdown(f"""
        <div class="result-card">
            <span class="result-rank">#{rank}</span>
            <div class="result-title">{doc_name}</div>
            <div class="result-score">📊 Skor: {score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
# Algorithm selector di pojok kanan atas
col_spacer, col_algo = st.columns([5, 1])
with col_algo:
    method = st.selectbox(
        "Algoritma:",
        ["TF-IDF", "BM25", "Bandingkan Keduanya"],
        index=0,
        label_visibility="visible"
    )

# Animated Tech Background
st.markdown("""
<div class="tech-background">
    <div class="grid-lines"></div>
    <div class="tech-line"></div>
    <div class="tech-line"></div>
    <div class="tech-line"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>
""", unsafe_allow_html=True)

# Header dengan title besar seperti Google
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">I N S I T E C H</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Document Search • AI-Powered Technology</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# SEARCH INTERFACE
# ==========================
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Query Input
query = st.text_input("", placeholder="🔍 Search for AI and technology documents...", key="search_query", label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# Trigger pencarian otomatis saat query berubah (user tekan Enter)
search_button = False
if query.strip() != "" and (query != st.session_state.last_query or method != st.session_state.last_method):
    search_button = True

# Initialize session state untuk pagination
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_method' not in st.session_state:
    st.session_state.last_method = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Reset halaman jika query atau method berubah
if search_button:
    if query != st.session_state.last_query or method != st.session_state.last_method:
        st.session_state.current_page = 1
    st.session_state.last_query = query
    st.session_state.last_method = method
    st.session_state.search_performed = True

# Konstanta untuk pagination
RESULTS_PER_PAGE = 10

# ==========================
# SEARCH RESULTS
# ==========================
# Gunakan query dan method dari session state jika ada
active_query = st.session_state.last_query if st.session_state.search_performed else query
active_method = st.session_state.last_method if st.session_state.search_performed else method

if search_button or st.session_state.search_performed:
    if active_query.strip() == "":
        st.warning("⚠️ Query tidak boleh kosong. Silakan masukkan kata kunci pencarian.")
        st.session_state.search_performed = False
    else:
        st.markdown("---")
        
        if active_method == "TF-IDF":
            st.markdown('<h2 class="section-header">Hasil Pencarian TF-IDF</h2>', unsafe_allow_html=True)
            
            # Ambil atau hitung hasil (semua hasil relevan)
            if search_button or st.session_state.search_results is None:
                all_results = search_tfidf(active_query)
                st.session_state.search_results = all_results
            else:
                all_results = st.session_state.search_results
            
            if all_results:
                # Hitung pagination
                total_pages = (len(all_results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
                current_page = st.session_state.current_page
                
                # Batasi halaman
                if current_page > total_pages:
                    st.session_state.current_page = 1
                    current_page = 1
                
                # Ambil hasil untuk halaman saat ini
                start_idx = (current_page - 1) * RESULTS_PER_PAGE
                end_idx = min(start_idx + RESULTS_PER_PAGE, len(all_results))
                page_results = all_results[start_idx:end_idx]
                
                # Info hasil
                st.markdown(f'<p class="page-info">Menampilkan hasil {start_idx + 1}-{end_idx} dari {len(all_results)} untuk "<strong>{active_query}</strong>"</p>', unsafe_allow_html=True)
                
                # Tampilkan hasil
                for idx, (doc, score) in enumerate(page_results):
                    rank = start_idx + idx + 1
                    display_result(doc, score, rank, active_query)
                
                # Pagination
                if total_pages > 1:
                    st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
                    
                    # Hitung range halaman yang ditampilkan (maksimal 5 halaman)
                    page_range = []
                    if total_pages <= 5:
                        page_range = list(range(1, total_pages + 1))
                    else:
                        if current_page <= 3:
                            page_range = [1, 2, 3, 4, 5]
                        elif current_page >= total_pages - 2:
                            page_range = list(range(total_pages - 4, total_pages + 1))
                        else:
                            page_range = list(range(current_page - 2, current_page + 3))
                    
                    # Tombol Previous
                    cols = st.columns([1] + [1]*len(page_range) + [1])
                    with cols[0]:
                        if current_page > 1:
                            if st.button("◄", key="prev_tfidf", type="secondary", use_container_width=True):
                                st.session_state.current_page = current_page - 1
                                st.rerun()
                        else:
                            st.button("◄", key="prev_tfidf_disabled", type="secondary", use_container_width=True, disabled=True)
                    
                    # Tombol nomor halaman
                    for idx, page_num in enumerate(page_range):
                        with cols[idx + 1]:
                            if st.button(str(page_num), key=f"page_tfidf_{page_num}", 
                                       type="primary" if page_num == current_page else "secondary",
                                       use_container_width=True):
                                st.session_state.current_page = page_num
                                st.rerun()
                    
                    # Tombol Next
                    with cols[-1]:
                        if current_page < total_pages:
                            if st.button("►", key="next_tfidf", type="secondary", use_container_width=True):
                                st.session_state.current_page = current_page + 1
                                st.rerun()
                        else:
                            st.button("►", key="next_tfidf_disabled", type="secondary", use_container_width=True, disabled=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Tidak ada hasil yang ditemukan.")
                st.session_state.search_performed = False

        elif active_method == "BM25":
            st.markdown('<h2 class="section-header">Hasil Pencarian BM25</h2>', unsafe_allow_html=True)
            
            # Ambil atau hitung hasil (semua hasil relevan)
            if search_button or st.session_state.search_results is None:
                all_results = search_bm25(active_query)
                st.session_state.search_results = all_results
            else:
                all_results = st.session_state.search_results
            
            if all_results:
                # Hitung pagination
                total_pages = (len(all_results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
                current_page = st.session_state.current_page
                
                # Batasi halaman
                if current_page > total_pages:
                    st.session_state.current_page = 1
                    current_page = 1
                
                # Ambil hasil untuk halaman saat ini
                start_idx = (current_page - 1) * RESULTS_PER_PAGE
                end_idx = min(start_idx + RESULTS_PER_PAGE, len(all_results))
                page_results = all_results[start_idx:end_idx]
                
                # Info hasil
                st.markdown(f'<p class="page-info">Menampilkan hasil {start_idx + 1}-{end_idx} dari {len(all_results)} untuk "<strong>{active_query}</strong>"</p>', unsafe_allow_html=True)
                
                # Tampilkan hasil
                for idx, (doc, score) in enumerate(page_results):
                    rank = start_idx + idx + 1
                    display_result(doc, score, rank, active_query)
                
                # Pagination
                if total_pages > 1:
                    st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
                    
                    # Hitung range halaman yang ditampilkan (maksimal 5 halaman)
                    page_range = []
                    if total_pages <= 5:
                        page_range = list(range(1, total_pages + 1))
                    else:
                        if current_page <= 3:
                            page_range = [1, 2, 3, 4, 5]
                        elif current_page >= total_pages - 2:
                            page_range = list(range(total_pages - 4, total_pages + 1))
                        else:
                            page_range = list(range(current_page - 2, current_page + 3))
                    
                    # Tombol Previous
                    cols = st.columns([1] + [1]*len(page_range) + [1])
                    with cols[0]:
                        if current_page > 1:
                            if st.button("◄", key="prev_bm25", type="secondary", use_container_width=True):
                                st.session_state.current_page = current_page - 1
                                st.rerun()
                        else:
                            st.button("◄", key="prev_bm25_disabled", type="secondary", use_container_width=True, disabled=True)
                    
                    # Tombol nomor halaman
                    for idx, page_num in enumerate(page_range):
                        with cols[idx + 1]:
                            if st.button(str(page_num), key=f"page_bm25_{page_num}", 
                                       type="primary" if page_num == current_page else "secondary",
                                       use_container_width=True):
                                st.session_state.current_page = page_num
                                st.rerun()
                    
                    # Tombol Next
                    with cols[-1]:
                        if current_page < total_pages:
                            if st.button("►", key="next_bm25", type="secondary", use_container_width=True):
                                st.session_state.current_page = current_page + 1
                                st.rerun()
                        else:
                            st.button("►", key="next_bm25_disabled", type="secondary", use_container_width=True, disabled=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Tidak ada hasil yang ditemukan.")
                st.session_state.search_performed = False

        else:
            st.markdown('<h2 class="section-header">Perbandingan TF-IDF vs BM25</h2>', unsafe_allow_html=True)
            
            # Reset search performed untuk mode perbandingan
            st.session_state.search_performed = False
            
            # Untuk mode perbandingan, tampilkan 10 hasil tanpa pagination
            col1, col2 = st.columns(2, gap="large")

            # TF-IDF Column
            with col1:
                st.markdown("### TF-IDF")
                r1 = search_tfidf(active_query, 10)
                if r1:
                    for rank, (doc, score) in enumerate(r1, 1):
                        display_result(doc, score, rank, active_query)
                else:
                    st.info("Tidak ada hasil.")

            # BM25 Column
            with col2:
                st.markdown("### BM25")
                r2 = search_bm25(active_query, 10)
                if r2:
                    for rank, (doc, score) in enumerate(r2, 1):
                        display_result(doc, score, rank, active_query)
                else:
                    st.info("Tidak ada hasil.")

st.markdown("""
<div class="footer-text">
    © 2025 INSITECH - Sistem Penelusuran Informasi Dokumen Teknologi dan AI
</div>
""", unsafe_allow_html=True)