import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ══════════════════════════════════════════════════════════════
#  API KEY SETUP (Gemini için)
# ══════════════════════════════════════════════════════════════
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ Streamlit Secrets içinde GOOGLE_API_KEY bulunamadı!")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="METU IE Summer Practice Chatbot", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #f8f5f0; }
.main-header { background: linear-gradient(90deg, #8B0000, #CC0000); padding: 1.4rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; box-shadow: 0 4px 16px rgba(139,0,0,0.25); }
.main-header h2 { color: white; margin: 0; font-size: 1.6rem; font-weight: 600; }
.main-header p  { color: #ffcccc; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE (Senin Q&A Veritabanın)
# ══════════════════════════════════════════════════════════════
QA_DATABASE = [
    {"question": "What documents are required for IE 300?", "answer": "Before: Application Form, Acceptance Letter, SGK Form. After: Logbook, Evaluation Form, Report."},
    {"question": "How do I apply for the internship?", "answer": "Go to sp-ie.metu.edu.tr, fill the form, upload documents, and wait for department approval."},
    {"question": "Can I do IE 300 and IE 400 in the same summer?", "answer": "No, you cannot. They must be done in separate summers."},
    {"question": "How can I find a summer practice? Company requirements?", "answer": "Look for manufacturing or service companies with at least 10 employees and 1 supervising engineer. Apply 2-3 months early."},
    # İstersen buraya kendi diğer sorularını da ekleyebilirsin
]

# ══════════════════════════════════════════════════════════════
#  1. YEREL ARAMA MOTORU (Sıfır 404 Hatası Garantili)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_vector_store():
    # Aramayı Google'a DEĞİL, senin bilgisayarına/sunucuya yaptırıyoruz!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    docs = [Document(page_content=qa["question"] + " " + qa["answer"]) for qa in QA_DATABASE]
    return FAISS.from_documents(docs, embeddings)

# ══════════════════════════════════════════════════════════════
#  2. GEMINI İLE CEVAPLAMA (Out of Scope Desteği)
# ══════════════════════════════════════════════════════════════
def ask_gemini(user_question: str) -> str:
    # Önce staj dökümanlarında bu soruya benzer bir şey var mı diye bak
    vector_store = build_vector_store()
    results = vector_store.similarity_search_with_score(user_question, k=2)
    
    context = ""
    # Eğer benzerlik çok iyi değilse (stajla alakasızsa), context boş kalır.
    # Eğer stajla alakalıysa contexti doldururuz.
    if results and results[0][1] < 1.7: 
        context = "\n".join([doc.page_content for doc, score in results])

    # Gemini'ye hem staj asistanı olduğunu hem de her şeye cevap verebileceğini söylüyoruz
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are an intelligent, friendly AI assistant for METU Industrial Engineering students.

    Here is the official internship context (if any):
    {context}

    Task Guidelines:
    1. If the user's question is about METU IE Internships (IE 300 / IE 400), answer accurately using ONLY the context provided above.
    2. OUT OF SCOPE REQUIREMENT: If the user asks something completely unrelated to internships (e.g., math, coding, general knowledge, recipes, jokes), DO NOT reject it! Answer it perfectly and politely using your general AI knowledge.

    User Question: {user_question}
    
    Answer:"""

    response = model.generate_content(prompt)
    return response.text

# ══════════════════════════════════════════════════════════════
#  UI — HEADER & SIDEBAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h2>🎓 METU IE Summer Practice Chatbot</h2>
    <p>IE 300 & IE 400 · Out of Scope Supported · Powered by Gemini</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Test Questions")
    if st.button("How to find a practice?"): st.session_state["prefill"] = "How can I find a summer practice?"
    if st.button("What is 15 * 24? (Out of scope)"): st.session_state["prefill"] = "What is 15 * 24?"
    if st.button("Write a Python hello world"): st.session_state["prefill"] = "Write a Python hello world code"
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  UI — CHAT
# ══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 I'm your METU IE Assistant. I can help you with your internship questions, or anything else you'd like to chat about!"}]

prefill = st.session_state.pop("prefill", None)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_gemini(user_input)
                st.markdown(answer)
            except Exception as e:
                st.error(f"⚠️ API Error: {e}")
                answer = "Error occurred."

    st.session_state.messages.append({"role": "assistant", "content": answer})
