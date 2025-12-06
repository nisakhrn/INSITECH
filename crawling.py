import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from urllib.parse import urljoin

TARGET_DOC = 500   # kamu bisa ubah bebas

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0"
}

# ================================
# SUMBER + pola pagination
# ================================
SOURCES = {
    "detikinet": "https://inet.detik.com/indeks/?page={}",
    "gadgetren": "https://gadgetren.com/page/{}",
    "cnbc": "https://www.cnbcindonesia.com/tech/indeks/{}",
    "liputan6": "https://www.liputan6.com/tekno?type=indeks&page={}",
    "antaranews": "https://www.antaranews.com/tekno/terkini/{}",
    "cnn": "https://www.cnnindonesia.com/teknologi/indeks/{}",
}

# relevansi dibuat lebih longgar
RELEVANT = [
    "ai","artificial","machine","learning","deep","robot","teknologi","chip",
    "data","digital","internet","software","hardware","cyber","gadget",
    "komputer","ponsel","smartphone","cloud","server"
]

def relevant(text):
    t = text.lower()
    return any(k in t for k in RELEVANT)

# ================================
# SAFE GET
# ================================
def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return None

# ================================
# Extract links
# ================================
def extract_links(base_url, page):
    url = base_url.format(page)
    html = safe_get(url)
    if not html:
        print(" - gagal load halaman")
        return []

    soup = BeautifulSoup(html, "html.parser")
    domain = base_url.split("//")[1].split("/")[0]

    links = set()

    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])

        # link harus ke domain yang sama
        if domain not in link:
            continue

        # jangan ambil URL daftar/tag
        if "/tag/" in link or "/search" in link or link.endswith("/"):
            continue

        # ambil hanya artikel (URL mengandung angka tanggal / ID)
        if any(c.isdigit() for c in link[-12:]):
            links.add(link)

    return list(links)

# ================================
# Extract Artikel
# ================================
def extract_article(url):
    html = safe_get(url)
    if not html:
        return None, None

    soup = BeautifulSoup(html, "html.parser")

    # judul umum
    title = soup.find("h1")
    title = title.get_text(strip=True) if title else None

    para = soup.find_all("p")
    content = "\n".join([p.get_text(strip=True) for p in para])

    if title and len(content) > 150:
        return title, content

    return None, None


# ================================
# MAIN
# ================================
docs = []
count = 0

print("\nMulai crawling...\n")

for name, url in SOURCES.items():

    print(f"\n=== Sumber: {name.upper()} ===")

    for page in range(1, 50):      # ambil 15 halaman per sumber !!
        if count >= TARGET_DOC:
            break

        print(f"Halaman {page}...")

        links = extract_links(url, page)
        print(f" - {len(links)} link ditemukan")

        for link in links:
            if count >= TARGET_DOC:
                break

            print(f"[{count+1}/{TARGET_DOC}] {link}")

            title, content = extract_article(link)
            if not title or not content:
                print("   - Skip (kosong/pendek)")
                continue

            if not relevant(title + " " + content):
                print("   - Skip (tidak relevan)")
                continue

            docs.append({
                "judul": title,
                "url": link,
                "isi": content
            })

            count += 1
            print("   - OK")

            time.sleep(0.5)

    print(f"Total dari sumber {name}: {count}")


# SAVE HASIL
df = pd.DataFrame(docs)
df.to_csv("korpus_500.csv", index=False, encoding="utf-8")

print("\n=== SELESAI ===")
print(f"Total dokumen terkumpul: {len(df)}")
print("Disimpan ke: korpus_500.csv")