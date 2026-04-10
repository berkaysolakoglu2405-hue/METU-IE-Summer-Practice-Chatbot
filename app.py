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
#  CSS (Gece ve Gündüz Modu Uyumlu)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] { 
    font-family: 'Inter', sans-serif; 
}
[data-testid="stSidebar"] { background: linear-gradient(180deg, #7a0000 0%, #4a0000 100%); }
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,0.12) !important; color: #fff !important; border: 1px solid rgba(255,255,255,0.25) !important; border-radius: 8px !important; text-align: left !important; width: 100% !important; margin-bottom: 4px !important; padding: 6px 10px !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.22) !important; }
.main-header { background: linear-gradient(90deg, #8B0000, #CC0000); padding: 1.4rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; box-shadow: 0 4px 16px rgba(139,0,0,0.25); }
.main-header h2 { color: white; margin: 0; font-size: 1.6rem; font-weight: 600; }
.main-header p  { color: #ffcccc; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

.footer { 
    text-align: center; 
    font-size: 0.75rem; 
    color: var(--text-color); 
    opacity: 0.6; 
    margin-top: 2rem; 
    padding-top: 1rem; 
    border-top: 1px solid rgba(128,128,128,0.2); 
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  Q&A DATABASE (Tam, 19 Soruluk Veritabanı)
# ══════════════════════════════════════════════════════════════
QA_DATABASE = [
    {
        "question": "What documents are required for IE 300? What papers do I need for IE 300 internship?",
        "answer": "**Required documents for IE 300:**\n\n**Before the internship:**\n- Internship Application Form (filled online at sp-ie.metu.edu.tr)\n- Company Acceptance Letter (signed by the company)\n- SGK Form (for social security — the department handles this)\n\n**After the internship:**\n- Internship Logbook (daily records, signed by supervisor)\n- Supervisor Evaluation Form (filled and signed by your company supervisor)\n- Internship Completion Report (20–40 pages)"
    },
    {
        "question": "What documents are required for IE 400? What papers do I need for IE 400 internship?",
        "answer": "**Required documents for IE 400:**\n\n**Before the internship:**\n- Internship Application Form (filled online at sp-ie.metu.edu.tr)\n- Company Acceptance Letter (signed by the company)\n- SGK Form (for social security — the department handles this)\n\n**After the internship:**\n- Internship Logbook (daily records, signed by supervisor)\n- Supervisor Evaluation Form (filled and signed by your company supervisor)\n- Internship Completion Report (20–40 pages)\n\nNote: You must have already completed IE 300 before you can start IE 400."
    },
    {
        "question": "What are the requirements for IE 300? IE 300 requirements prerequisites conditions",
        "answer": "**IE 300 (Industrial Training I) requirements:**\n\n- You must have successfully completed **IE 200** before applying\n- Minimum duration: **20 working days (4 weeks)**\n- Must be done at a manufacturing or service sector company\n- The company must have at least 10 employees and a supervising engineer\n- You must get departmental approval before starting\n- Focus is on basic engineering applications and hands-on observation"
    },
    {
        "question": "What are the requirements for IE 400? IE 400 requirements prerequisites conditions",
        "answer": "**IE 400 (Industrial Training II) requirements:**\n\n- You must have successfully completed **IE 300** before applying\n- Minimum duration: **20 working days (4 weeks)**\n- Focus is on advanced engineering: system design, process improvement, engineering analysis\n- Same company eligibility rules as IE 300 apply\n- IE 300 and IE 400 **cannot** be done in the same summer"
    },
    {
        "question": "How do I apply for
