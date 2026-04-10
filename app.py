import streamlit as st
import streamlit.components.v1 as components
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
#  CSS (Orijinal Bordo Tema ve Detaylar)
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
#  Q&A DATABASE (Aynı Veritabanı)
# ══════════════════════════════════════════════════════════════
QA_DATABASE = [
    {"question": "What documents are required for IE 300? What papers do I need for IE 300 internship?", "answer": "**Required documents for IE 300:**\n\n- Internship Application Form (sp-ie.metu.edu.tr)\n- Company Acceptance Letter\n- SGK Form\n- Logbook, Supervisor Evaluation, and Report (Afterwards)"},
    {"question": "What documents are required for IE 400? What papers do I need for IE 400 internship?", "answer": "**Required documents for IE 400:**\n\n- Internship Application Form (sp-ie.metu.edu.tr)\n- Company Acceptance Letter\n- SGK Form\n- Logbook, Supervisor Evaluation, and Report (Afterwards)\n- **PREREQUISITE:** You must have already completed IE 300 before you can start IE 400."},
    {"question": "What are the requirements for IE 300? IE 300 requirements prerequisites conditions", "answer": "**IE 300 (Industrial Training I) requirements:**\n- Successfully completed **IE 200**\n- Min duration: 20 working days\n- Company must have min 10 employees and 1 supervising engineer."},
    {"question": "What are the requirements for IE 400? IE 400 requirements prerequisites conditions", "answer": "**IE 400 (Industrial Training II) requirements:**\n- Successfully completed **IE 300**\n- Min duration: 20 working days\n- Company must have min 10 employees and 1 supervising engineer."},
    {"question": "How do I apply for the internship? How to apply summer practice steps", "answer": "1. Fill form at sp-ie.metu.edu.tr\n2. Get Acceptance Letter\n3. Wait for department approval (Apply at least 3 weeks before start)."},
    {"question": "How long does the internship last? Duration weeks days", "answer": "Min 20 working days. Weekends and holidays do NOT count."},
    {"question": "Can I do IE 300 and IE 400 in the same summer?", "answer": "No. IE 300 and IE 400 cannot be done in the same summer."},
    {"question": "What is the difference between IE 300 and IE 400?", "answer": "IE 300: Basic engineering, observation. Req: IE 200.\nIE 400: Advanced analysis, design. Req: IE 300."},
    {"question": "Can I do my internship abroad?", "answer": "Yes, but you need prior departmental approval."},
    {"question": "Who arranges the SGK insurance?", "answer": "The department secretary handles it after your documents are approved."},
    {"question": "Will I be paid during my internship?", "answer": "Payment depends on the company; it is not mandatory."},
    {"question": "What companies are eligible for internship?", "answer": "Manufacturing or service companies with min 10 employees and 1 engineer."},
    {"question": "How should I fill the internship logbook?", "answer": "Fill daily, get signed by supervisor, submit at the end."},
    {"question": "How should I prepare the internship report?", "answer": "English or Turkish, 20-40 pages, submit after the internship."},
    {"question": "How is the internship evaluated?", "answer": "Logbook + Report + Supervisor Form."},
    {"question": "What happens if my internship is not approved?", "answer": "You might need to make corrections or redo the practice."},
    {"question": "When is the application deadline?", "answer": "Usually in May or June. Check sp-ie.metu.edu.tr for updates."},
    {"question": "Where can I find official forms information announcements?", "answer": "Visit https://sp-ie.metu.edu.tr/en"},
    {"question": "How can I find a summer practice?", "answer": "Start early, check LinkedIn, Career Fairs, and your network."}
]

# ══════════════════════════════════════════════════════════════
#  YEREL ARAMA MOTORU
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading knowledge base...")
def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    docs = [Document(page_content=qa["question"] + " " + qa["answer"], metadata={"answer": qa["answer"]}) for qa in QA_DATABASE]
    return FAISS.from_documents(docs, embeddings)

# ══════════════════════════════════════════════════════════════
#  KİMLİĞİ GİZLENMİŞ, KOTA KORUMALI VE DOĞRU BEYİN
# ══════════════════════════════════════════════════════════════
def ask_gemini(user_question: str) -> str:
    vector_store = build_vector_store()
    results = vector_store.similarity_search(user_question, k=3)
    context = "\n\n---\n\n".join([doc.metadata["answer"] for doc in results])

    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        safe_models = [m for m in available_models if "2.0" not in m and "2.5" not in m and "vision" not in m]
        chosen_model = next((m for m in safe_models if "1.5-flash" in m), safe_models[0]).replace("models/", "")
        model = genai.GenerativeModel(chosen_model)
    except Exception as e:
        return f"API bağlantı hatası: {e}"
    
    prompt = f"""You are the official 'METU IE Summer Practice Assistant'.
    
    CRITICAL IDENTITY RULE: 
    Act like a helpful human-like assistant dedicated to METU IE students.
    
    OFFICIAL INFORMATION FROM DATABASE:
    {context}
    
    STRICT GUIDELINES:
    1. Base your answer ONLY on the OFFICIAL INFORMATION above. 
    2. DO NOT hallucinate. For example: IE 400 requires IE 300, NOT IE 200. Check the data carefully.
    3. If the answer is not in the context, say "I couldn't find the exact information in our records. Please check sp-ie.metu.edu.tr."
    4. Keep the helpful, conversational tone of an assistant.

    User Question: {user_question}
    Answer:"""

    response = model.generate_content(prompt)
    return response.text

# ══════════════════════════════════════════════════════════════
#  UI (Orijinal Arayüz)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h2>🎓 METU IE Summer Practice Chatbot</h2>
    <p>IE 300 &amp; IE 400 · Applications · Documents · Deadlines</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 METU IE\nSummer Practice")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 I'm the **METU IE Summer Practice Assistant**. How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about summer practice...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_gemini(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

components.html(
    f"""
    <script>
        var chatHistory = window.parent.document.querySelector('.main');
        if (chatHistory) {{
            chatHistory.scrollTo({{ top: chatHistory.scrollHeight, behavior: 'smooth' }});
        }}
    </script>
    """,
    height=0
)
