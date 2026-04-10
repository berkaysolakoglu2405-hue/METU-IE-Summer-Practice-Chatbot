import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="METU IE Summer Practice Chatbot",
    page_icon="🎓",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
#  API KEY SETUP
# ══════════════════════════════════════════════════════════════
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ Streamlit Secrets içinde GOOGLE_API_KEY bulunamadı!")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  CSS (Orijinal Bordo Tema)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #f8f5f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #7a0000 0%, #4a0000 100%); }
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,0.12) !important; color: #fff !important; border: 1px solid rgba(255,255,255,0.25) !important; border-radius: 8px !important; text-align: left !important; width: 100% !important; margin-bottom: 4px !important; padding: 6px 10px !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.22) !important; }
.main-header { background: linear-gradient(90deg, #8B0000, #CC0000); padding: 1.4rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; box-shadow: 0 4px 16px rgba(139,0,0,0.25); }
.main-header h2 { color: white; margin: 0; font-size: 1.6rem; font-weight: 600; }
.main-header p  { color: #ffcccc; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
.footer { text-align: center; font-size: 0.75rem; color: #aaa; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0d8d0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  Q&A DATABASE (Senin Veritabanın)
# ══════════════════════════════════════════════════════════════
QA_DATABASE = [
    {"question": "What documents are required for IE 300? What papers do I need for IE 300 internship?", "answer": "**Required documents for IE 300:**\n\n**Before the internship:**\n- Internship Application Form (filled online at sp-ie.metu.edu.tr)\n- Company Acceptance Letter (signed by the company)\n- SGK Form (for social security — the department handles this)\n\n**After the internship:**\n- Internship Logbook (daily records, signed by supervisor)\n- Supervisor Evaluation Form (filled and signed by your company supervisor)\n- Internship Completion Report (20–40 pages)"},
    {"question": "What documents are required for IE 400? What papers do I need for IE 400 internship?", "answer": "**Required documents for IE 400:**\n\n**Before the internship:**\n- Internship Application Form (filled online at sp-ie.metu.edu.tr)\n- Company Acceptance Letter (signed by the company)\n- SGK Form (for social security — the department handles this)\n\n**After the internship:**\n- Internship Logbook (daily records, signed by supervisor)\n- Supervisor Evaluation Form (filled and signed by your company supervisor)\n- Internship Completion Report (20–40 pages)\n\nNote: You must have already completed IE 300 before you can start IE 400."},
    {"question": "How do I apply for the internship? How to apply summer practice application process steps", "answer": "**How to apply for the internship:**\n\n1. Go to **sp-ie.metu.edu.tr** and fill the online application form\n2. Enter your company details and planned internship dates\n3. Upload required documents (acceptance letter, etc.)\n4. Submit and wait for **department approval**\n5. After approval, the department will handle SGK (social security) registration\n\n⚠️ Apply **at least 3 weeks** before your internship start date."},
    {"question": "How can I find a summer practice? How to find a company for internship? Find internship place", "answer": "**How to find a company for your summer practice:**\n\n- Start searching **early** — at least 2-3 months before the internship period\n- Look for companies in IE-related sectors: manufacturing, logistics, banking, hospitals, consulting, IT\n- The company must have **at least 10 employees** and an engineer who can supervise you\n- Check METU career fairs, LinkedIn, company websites, and your personal network\n- Once a company agrees, get their **Acceptance Letter** and apply through sp-ie.metu.edu.tr\n- You can intern **abroad** too — just get departmental approval first"}
]

# ══════════════════════════════════════════════════════════════
#  YEREL ARAMA MOTORU
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading knowledge base… (~30 seconds on first load)")
def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    docs = [Document(page_content=qa["question"] + " " + qa["answer"], metadata={"answer": qa["answer"], "idx": i}) for i, qa in enumerate(QA_DATABASE)]
    return FAISS.from_documents(docs, embeddings)

# ══════════════════════════════════════════════════════════════
#  YENİ: KENDİ KENDİNİ TAMİR EDEN MELEZ BEYİN
# ══════════════════════════════════════════════════════════════
def ask_gemini(user_question: str) -> str:
    vector_store = build_vector_store()
    results = vector_store.similarity_search_with_score(user_question, k=2)
    
    context = ""
    if results and results[0][1] < 1.8: 
        context = "\n\n".join([doc.metadata["answer"] for doc, score in results])

    # 1. Google'dan "Senin elinde benim için hangi modeller var?" diye listeyi çek
    available_models = []
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
        return f"API bağlantı hatası. Google API Key'iniz geçersiz veya servis kapalı olabilir: {e}"
        
    if not available_models:
        return "⚠️ HATA: Girdiğiniz API Key'in bağlı olduğu projede hiçbir Gemini modeline erişim yetkisi yok! (Google Cloud üzerinden Generative Language API'yi aktif ettiğinizden emin olun)."

    # 2. Listeden en mantıklı modeli bul (Flash varsa onu al, yoksa listedeki İLK bulduğu çalışan modeli al)
    chosen_model = available_models[0] # Varsayılan olarak listedeki ilk modeli al
    for m in available_models:
        if "gemini-1.5-flash" in m:
            chosen_model = m
            break
        elif "gemini-pro" in m:
            chosen_model = m

    # 3. Model isminin başındaki 'models/' kısmını temizle (kütüphane kendi eklediği için çakışıyor olabilir)
    clean_model_name = chosen_model.replace("models/", "")
    
    # Sisteme haber veriyoruz hangi modeli seçtiğini (Streamlit ekranında ufak görünecek)
    st.toast(f"🤖 Bot '{clean_model_name}' modelini kullanarak cevap veriyor...", icon="✅")

    model = genai.GenerativeModel(clean_model_name)
    
    prompt = f"""You are an intelligent, friendly AI assistant for METU Industrial Engineering students.
    Here is the official internship context (if any):
    {context}
    Task Guidelines:
    1. If the user's question is about METU IE Internships, answer accurately using ONLY the context provided above.
    2. OUT OF SCOPE REQUIREMENT: If the user asks something completely unrelated to internships, DO NOT reject it! Answer it perfectly and politely using your general AI knowledge.
    User Question: {user_question}
    Answer:"""

    response = model.generate_content(prompt)
    return response.text

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.markdown("""<div class="main-header"><h2>🎓 METU IE Summer Practice Chatbot</h2><p>IE 300 &amp; IE 400 · Out of Scope Supported</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 METU IE\nSummer Practice")
    st.markdown("---")
    st.markdown("### 📚 Sample Questions")
    samples = ["What documents are required for IE 300?", "How do I apply for the internship?", "How can I find a summer practice?"]
    for q in samples:
        if st.button(q, key=f"btn_{q[:15]}"):
            st.session_state["prefill"] = q
    st.markdown("---")
    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 I'm the **METU IE Summer Practice Assistant**.\n\nI can answer your questions about **IE 300** and **IE 400** internships, or we can chat about anything else you'd like! 🚀"}]

prefill = st.session_state.pop("prefill", None)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about METU IE Summer Practice or general topics…") or prefill

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
                st.error(f"⚠️ Error: {e}")
                answer = "An error occurred."

    st.session_state.messages.append({"role": "assistant", "content": answer})
