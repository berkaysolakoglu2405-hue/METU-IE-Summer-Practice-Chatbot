import streamlit as st
import google.generativeai as genai
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# 1. API KEY'İ SİSTEMİN DAMARLARINA SOKUYORUZ
if "GOOGLE_API_KEY" in st.secrets:
    key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = key # Sistem değişkeni olarak ata
    genai.configure(api_key=key)        # Google kütüphanesini ayarla
else:
    st.error("⚠️ Secrets kısmında 'GOOGLE_API_KEY' bulunamadı!")
    st.stop()

st.set_page_config(page_title="METU IE Summer Practice Chatbot", page_icon="🎓")

# 2. EMBEDDING FONKSİYONU - ANAHTARI İÇİNE ZORLA SOKUYORUZ
@st.cache_resource(show_spinner=False)
def build_vector_store():
    # Buradaki api_key parametresini sildim, direkt os.environ'dan okutacağız
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ["GOOGLE_API_KEY"] # Zorla buraya yazıyoruz
    )
    
    # Senin dökümanların (Burayı kendi metninle doldur)
    KNOWLEDGE_BASE = "STAJ BILGILERI BURAYA..." 
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(KNOWLEDGE_BASE)
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embeddings)

# 3. CEVAP VERME FONKSİYONU
def ask_gemini(question: str, context: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Context: {context}\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text

# --- UI KISMI ---
st.title("🎓 METU IE Intern Assistant")

if user_input := st.chat_input("Sorunu buraya yaz..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        try:
            # Embedding burada yapılıyor, hata buradaysa yakalayacağız
            vector_store = build_vector_store()
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])
            answer = ask_gemini(user_input, context)
            st.markdown(answer)
        except Exception as e:
            st.error(f"❌ HATA: {e}")
