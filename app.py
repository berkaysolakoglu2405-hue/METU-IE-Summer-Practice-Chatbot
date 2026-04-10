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
#  Q&A DATABASE (Aynı Veriler)
# ══════════════════════════════════════════════════════════════
QA_DATABASE = [
    {"question": "What documents are required for IE 300? What papers do I need for IE 300 internship?", "answer": "**Required documents for IE 300:**\n\n- Internship Application Form (sp-ie.metu.edu.tr)\n- Company Acceptance Letter\n- SGK Form\n- Logbook, Supervisor Evaluation, and Report (Afterwards)"},
    {"question": "What documents are required for IE 400? What papers do I need for IE 400 internship?", "answer": "**Required documents for IE 400:**\n\n- Internship Application Form (sp-ie.metu.edu.tr)\n- Company Acceptance Letter\n- SGK Form\n- Logbook, Supervisor Evaluation, and Report (Afterwards)\n- PREREQUISITE: Must have completed IE 300."},
    {"question": "What are the requirements for IE 300? IE 300 requirements prerequisites conditions", "answer": "**IE 300 requirements:**\n- Completed **IE 200**\n- Min 20 working days\n- Min 10 employees + 1 Engineer supervisor."},
    {"question": "What are the requirements for IE 400? IE 400 requirements prerequisites conditions", "answer": "**IE 400 requirements:**\n- Completed **IE 300**\n- Min 20 working days\n- Min 10 employees + 1 Engineer supervisor."},
    {"question": "How do I apply for the internship? How to apply steps", "answer": "1. Fill form at sp-ie.metu.edu.tr\n2. Get Acceptance Letter\n3. Wait for department approval (Apply 3 weeks before start)."},
    {"question": "How long does the internship last? Duration days", "answer": "Min 20 working days for both IE 300 and IE 400. Weekends don't count."},
    {"question": "Can I do IE 300 and IE 400 in the same summer?", "answer": "No. They must be done in separate summers."},
    {"question": "What is the difference between IE 300 and IE 400?", "answer": "IE 300 (Training I): Basic engineering & observation. Req: IE 200.\nIE 400 (Training II): Advanced analysis & design. Req: IE 300."},
    {"question": "Can I do my internship abroad?", "answer": "Yes, with prior department approval."},
    {"question": "Who arranges the SGK insurance?", "answer": "The IE Department secretary handles it after your approval."},
    {"question": "Will I be paid during my internship?", "answer": "Depends on the company; it's not mandatory."},
    {"question": "What companies are eligible for internship?", "answer": "Manufacturing or service, min 10 employees, min 1 engineer."},
    {"question": "How should I fill the internship logbook?", "answer": "Fill daily, get signed by supervisor, submit at the end."},
    {"question": "How should I prepare the internship report?", "answer": "English/Turkish, 20-40 pages, submit after training."},
    {"question": "How is the internship evaluated?", "answer": "Logbook + Report + Supervisor Form."},
    {"question": "What happens if my internship is not approved?", "answer": "You might need to make corrections or redo it."},
    {"question": "When is the application deadline?", "answer": "Usually May or June. Check sp-ie.metu.edu.tr for exact dates."},
    {"question": "Where can I find official forms?", "answer": "Visit https://sp-ie.metu.edu.tr/en"},
    {"question": "How can I find a summer practice?", "answer": "Start early, check LinkedIn, Career Fairs, and personal network."}
]

# ══════════════════════════════════════════════════════════════
#  YEREL ARAMA MOTORU
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading...")
def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    docs = [Document(page_content=qa["question"] + " " + qa["answer"], metadata={"answer": qa["answer"]}) for qa in QA_DATABASE]
    return FAISS.from_documents(docs, embeddings)

# ══════════════════════════════════════════════════════════════
#  DOĞRUCU MELEZ BEYİN
# ══════════════════════════════════════════════════════════════
def ask_gemini(user_question: str) -> str:
    vector_store = build_vector_store()
    results = vector_store.similarity_search(user_question, k=3)
    context = "\n\n---\n\n".join([doc.metadata["answer"] for doc in results])

    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        safe_models = [m for m in available_models if "2.0" not in m and "2.5" not in m]
        chosen_model = next((m for m in safe_models if "1.5-flash" in m), safe_models[0]).replace("models/", "")
        model = genai.GenerativeModel(chosen_model)
    except Exception as e:
        return f"Error: {e}"
    
    # 🚨 KRİTİK GÖTTEN SIKMA ÖNLEYİCİ PROMPT 🚨
    prompt = f"""You are the official 'METU IE Summer Practice Assistant'. 
    
    IDENTITY: Human-like, helpful METU assistant.
    
    OFFICIAL DATA (STRICT):
    {context}
    
    RULES:
    1. For internship questions, ONLY use the 'OFFICIAL DATA' above. 
    2. TRUTH TEST: In the context, IE 400 requires IE 300. IE 300 requires IE 200. NEVER say IE 400 requires IE 200.
    3. If the answer is not in the data, say "I couldn't find this in the records, please check with the secretary."
    4. For general chat, stay in character but don't invent METU rules.

    User Question: {user_question}
    Answer:"""

    response = model.generate_content(prompt)
    return response.text

# ══════════════════════════════════════════════════════════════
#  UI (Orijinal Akış)
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-header"><h2>🎓 METU IE Summer Practice Chatbot</h2></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 I'm the METU IE Assistant. How can I help?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = ask_gemini(user_input)
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

components.html("""<script>var c = window.parent.document.querySelector('.main'); if(c) c.scrollTo({top: c.scrollHeight, behavior:'smooth'});</script>""", height=0)
