# 🎓 METU IE Yaz Stajı Chatbot

**IE 304 – Project 1** | RAG Pipeline · LangChain · FAISS · Google Gemini · Streamlit

---

## 📁 Dosya Yapısı

```
metu-ie-chatbot/
├── app.py              ← Ana Streamlit uygulaması
├── faq_data.py         ← 15 sık sorulan soru (sabit veri)
├── scrape.py           ← Web sitesini çeken script
├── requirements.txt    ← Python paket listesi
└── README.md           ← Bu dosya
```

---

## 🔑 ADIM 1 — Google Gemini API Key Al (Ücretsiz)

> Bu adım yaklaşık **2 dakika** sürer.

1. Tarayıcında şu adresi aç → **https://aistudio.google.com/app/apikey**
2. Google hesabınla giriş yap (Gmail hesabın varsa hazırsın)
3. Sayfanın ortasında **"Create API Key"** butonuna tıkla
4. Çıkan pencereden **"Create API key in new project"** seç
5. `AIzaSy...` ile başlayan bir kod göreceksin — **kopyala ve bir yere kaydet**

> ⚠️ Bu key'i başkalarıyla paylaşma!

---

## 💻 ADIM 2 — Python Ortamını Kur (Bilgisayarında)

### Python yüklü mü kontrol et

Terminali (Komut İstemi) aç ve şunu yaz:

```bash
python --version
```

Eğer `Python 3.10` veya üstü çıkıyorsa devam et.
Çıkmıyorsa → **https://www.python.org/downloads/** adresinden Python 3.11'i indir ve kur.

### Paketleri yükle

```bash
pip install -r requirements.txt
```

> Bu komut birkaç dakika sürebilir, bekle.

---

## 🕷️ ADIM 3 — Siteyi Scrape Et

```bash
python scrape.py
```

Bu komut sp-ie.metu.edu.tr sitesini gezerek tüm içeriği
`scraped_data.txt` dosyasına kaydeder.

> ⏱️ Yaklaşık 2–5 dakika sürer.
> Terminalde her sayfanın `[OK]` ile onaylandığını göreceksin.

**Not:** Scrape çalışmazsa (ağ engeli vs.) chatbot yine de çalışır —
FAQ verisiyle devam eder.

---

## 🚀 ADIM 4 — Uygulamayı Lokal Çalıştır (Test İçin)

```bash
streamlit run app.py
```

Tarayıcın otomatik olarak `http://localhost:8501` adresini açacak.
Sol menüdeki **API Key** kutusuna Gemini key'ini gir, sohbete başla!

---

## ☁️ ADIM 5 — GitHub'a Yükle

> GitHub hesabın yoksa → **https://github.com/join** adresinden ücretsiz aç.

### 5a. Yeni repository oluştur

1. **https://github.com/new** adresine git
2. **Repository name:** `metu-ie-chatbot` yaz
3. **Public** seç (Streamlit Cloud için gerekli)
4. **"Create repository"** butonuna tıkla

### 5b. Dosyaları yükle

Açılan sayfada **"uploading an existing file"** linkine tıkla.

Şu dosyaları sürükle-bırak yap:
- `app.py`
- `faq_data.py`
- `scrape.py`
- `requirements.txt`
- `scraped_data.txt` *(scrape.py çalıştıysa)*

**"Commit changes"** butonuna tıkla.

---

## 🌐 ADIM 6 — Streamlit Community Cloud'a Deploy Et

> **Tamamen ücretsiz!** Kredi kartı gerekmez.

### 6a. Streamlit Cloud hesabı aç

1. **https://share.streamlit.io** adresine git
2. **"Continue with GitHub"** butonuna tıkla
3. GitHub hesabınla giriş yap ve izinleri onayla

### 6b. Uygulamayı deploy et

1. Giriş yaptıktan sonra **"New app"** butonuna tıkla
2. Karşına 3 kutulu bir form çıkacak:
   - **Repository:** `kullanici-adin/metu-ie-chatbot` seç (açılır menüden)
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. **"Advanced settings"** butonuna tıkla
4. **Secrets** bölümüne şunu yapıştır (kendi key'ini yaz):
   ```toml
   GOOGLE_API_KEY = "AIzaSy_senin_key_in_buraya"
   ```
5. **"Deploy!"** butonuna tıkla

> ⏳ İlk deploy 3–5 dakika sürer. Yeşil `Running` yazısı çıkınca hazır!

### 6c. URL'ni paylaş

Deploy tamamlandığında şuna benzer bir URL alacaksın:

```
https://kullanici-adin-metu-ie-chatbot-app-xxxx.streamlit.app
```

Bu URL'yi tarayıcıdan doğrudan açılır — kurulum gerektirmez! ✅

---

## ❓ Sık Karşılaşılan Sorunlar

| Hata | Çözüm |
|------|-------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` tekrar çalıştır |
| `API key invalid` | Key'i doğru kopyaladığından emin ol, başında/sonunda boşluk olmasın |
| `FAISS error` | `faiss-cpu` paketini tekrar yükle: `pip install faiss-cpu` |
| Streamlit Cloud'da hata | Secrets kısmında key'i doğru girdiğini kontrol et |
| `scraped_data.txt` yok | Sadece FAQ verisiyle çalışır, sorun değil |

---

## 🏗️ Sistem Mimarisi

```
Kullanıcı Sorusu
       ↓
[Kapsam Dışı Filtre]  ──→  Staj dışıysa: kibarca reddet
       ↓
[FAISS Retriever]  ←── scraped_data.txt + FAQ verileri
       ↓
  En alakalı 5 chunk
       ↓
[Google Gemini 2.0 Flash] + RAG Prompt
       ↓
  Güvenilir Cevap → Kullanıcı
```

### Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Arayüz | Streamlit |
| Vektör DB | FAISS (faiss-cpu) |
| Embedding | Google `embedding-001` |
| LLM | Gemini 2.0 Flash |
| Orchestration | LangChain |
| Veri Kaynağı | sp-ie.metu.edu.tr + özel FAQ |

---

## 📋 Test Soruları (Ödev İçin)

1. **"IE 300 stajı için hangi belgeler gereklidir?"**
2. **"Yurt dışında staj yapabilir miyim?"**
3. **"Staj defteri nasıl doldurulmalıdır?"**
4. **"IE 300 ve IE 400 aynı yaz yapılabilir mi?"**
5. **"SGK sigortası nasıl yapılır?"**

---

*METU IE 304 – Project 1 | Intelligent Chatbot Application*
