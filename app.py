import streamlit as st
import google.generativeai as genai
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="METU IE Summer Practice Chatbot", page_icon="🎓")

# ── API Key Setup (Streamlit Secrets'tan çekiyoruz) ────────────
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ API Key bulunamadı! Lütfen Streamlit Cloud Settings > Secrets kısmına anahtarı ekleyin.")
    st.stop()

# ── CSS Theme ──────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #8B0000, #CC0000);
    padding: 1.2rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 1rem;
}
.main-header h2 {color: white; margin: 0;}
.main-header p {color: #ffcccc; margin: 0.3rem 0 0 0; font-size: 0.9rem;}
.footer {text-align: center; font-size: 0.75rem; color: #888; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ── Knowledge Base (Burayı senin metninle aynı bıraktım) ────────
KNOWLEDGE_BASE = """
=== METU IE SUMMER PRACTICE - OFFICIAL INFORMATION ===
(Buraya senin paylaştığın uzun staj dökümanı gelecek...)
"""

FAQ_TEXT = """
(Buraya senin paylaştığın SSS metni gelecek...)
"""

# ── Out-of-scope detection ─────────────────────────────────────
SCOPE_KEYWORDS = ["internship", "staj", "ie 300", "ie 400", "metu", "odtu"] # vs. listeyi uzatabilirsin

def is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return not any(kw in q for kw in SCOPE_KEYWORDS)

# ── Build FAISS vector store ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vector_store(api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # Stabil ve yeni model
        google_api_key=api_key,
    )

    full_text = KNOWLEDGE_BASE + "\n\n" + FAQ_TEXT
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=c) for c in chunks]

    return FAISS.from_documents(docs, embeddings)

# ── Ask Gemini ───────────────────────────────────────────────
def ask_gemini(api_key: str, question: str, context: str) -> str:
    genai.configure(api_key=api_key)
    # 1.5-flash en stabil olanıdır, kredin varken canavar gibi çalışır
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""You are a helpful assistant for METU Industrial Engineering Summer Practice.
Answer the student's question using ONLY the context below.
If the answer is not in the context, say: "For the most accurate information, please visit sp-ie.metu.edu.tr."

Context:
{context}

Question: {question}

Answer:"""

    response = model.generate_content(prompt)
    return response.text

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h2>🎓 METU IE Summer Practice Chatbot</h2>
    <p>IE 300 & IE 400 · Applications · Documents · Deadlines</p>
</div>
""", unsafe_allow_html=True)

# Sidebar (Artık kutucuk yok, sadece temizleme butonu)
with st.sidebar:
    st.markdown("### ⚙️ Dashboard")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("📌 **Source:** [sp-ie.metu.edu.tr](https://sp-ie.metu.edu.tr/en)")

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 How can I help you with your METU IE internship?"}]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask about METU IE Summer Practice..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_out_of_scope(user_input):
            answer = "🚫 This question is outside the scope of METU IE Summer Practice."
            st.markdown(answer)
        else:
            with st.spinner("Searching knowledge base..."):
                try:
                    vector_store = build_vector_store(api_key)
                    docs = vector_store.similarity_search(user_input, k=5)
                    context = "\n\n".join([d.page_content for d in docs])
                    answer = ask_gemini(api_key, user_input, context)
                    st.markdown(answer)
                except Exception as e:
                    answer = f"⚠️ Error: `{e}`"
                    st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
