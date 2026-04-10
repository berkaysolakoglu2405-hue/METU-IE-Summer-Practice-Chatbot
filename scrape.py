"""
scrape.py
---------
sp-ie.metu.edu.tr sitesindeki tüm sayfaları scrape eder
ve içerikleri 'scraped_data.txt' dosyasına kaydeder.

Kullanım:
    python scrape.py
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

BASE_URL = "https://sp-ie.metu.edu.tr/en"
DOMAIN   = "sp-ie.metu.edu.tr"
OUTPUT   = "scraped_data.txt"

visited = set()
all_text = []

def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)

def scrape_page(url: str):
    if url in visited:
        return
    visited.add(url)

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            print(f"  [SKIP] {url}  →  HTTP {resp.status_code}")
            return
    except Exception as e:
        print(f"  [ERROR] {url}  →  {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Sayfanın okunabilir metnini al
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = clean_text(text)

    if text:
        all_text.append(f"\n\n{'='*60}\nKAYNAK URL: {url}\n{'='*60}\n{text}")
        print(f"  [OK] {url}  ({len(text)} karakter)")

    # Aynı domain'deki linkleri bul
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(url, href)
        parsed = urlparse(full)
        if parsed.netloc == DOMAIN and full not in visited:
            scrape_page(full)
            time.sleep(0.5)   # sunucuya nazik ol

def main():
    print(f"Scraping başlıyor: {BASE_URL}\n")
    scrape_page(BASE_URL)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    print(f"\n✅ Tamamlandı! {len(visited)} sayfa tarandı.")
    print(f"📄 Çıktı dosyası: {OUTPUT}")

if __name__ == "__main__":
    main()
