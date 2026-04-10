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
#  Q&A DATABASE (Senin Tam, 19 Soruluk Veritabanın!)
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
        "question": "How do I apply for the internship? How to apply summer practice application process steps",
        "answer": "**How to apply for the internship:**\n\n1. Go to **sp-ie.metu.edu.tr** and fill the online application form\n2. Enter your company details and planned internship dates\n3. Upload required documents (acceptance letter, etc.)\n4. Submit and wait for **department approval**\n5. After approval, the department will handle SGK (social security) registration\n\n⚠️ Apply **at least 3 weeks** before your internship start date."
    },
    {
        "question": "How long does the internship last? Duration weeks days length of internship",
        "answer": "**Internship duration:**\n\n- Both IE 300 and IE 400 require a minimum of **20 working days (≈ 4 calendar weeks)**\n- Weekends and official public holidays do **NOT** count toward the 20 days\n- Any leave days granted by the company are also deducted from the total\n- You must complete the full 20 working days for the internship to be valid"
    },
    {
        "question": "Can I do IE 300 and IE 400 in the same summer? Both internships same time simultaneously",
        "answer": "**No, you cannot do IE 300 and IE 400 in the same summer.**\n\n- IE 300 must be fully completed and officially approved before you can begin IE 400\n- They must be completed in **separate summers** (or separate terms)\n- There are no exceptions to this rule"
    },
    {
        "question": "What is the difference between IE 300 and IE 400? Compare IE 300 IE 400",
        "answer": "**IE 300 vs IE 400:**\n\n| | IE 300 | IE 400 |\n|---|---|---|\n| Who | 3rd-year students | 4th-year students |\n| Prerequisite | IE 200 completed | IE 300 completed |\n| Focus | Basic engineering, observation | Advanced problems, system design |\n| Duration | Min. 20 working days | Min. 20 working days |\n\nIE 300 emphasises learning through observation and basic tasks.\nIE 400 expects you to contribute to engineering analysis, process improvement, or system design."
    },
    {
        "question": "Can I do my internship abroad? International internship foreign country outside Turkey",
        "answer": "**Yes, internships abroad are allowed!**\n\n- You must get **prior approval** from the IE Department before starting\n- The same requirements apply: minimum 20 working days, same documents, same report\n- For **Erasmus+ internship funding**, contact the METU International Relations Office\n- Apply well in advance — international approvals may take longer"
    },
    {
        "question": "Who arranges the SGK insurance? Social security registration internship insurance",
        "answer": "**SGK (Social Security) insurance is arranged by the IE Department secretary** — you do not handle it yourself.\n\n- Submit all your required documents to the department **before** your internship starts\n- Students are covered by occupational accident and occupational disease insurance during the internship\n- Do **not** start your internship before SGK registration is confirmed"
    },
    {
        "question": "Will I be paid during my internship? Salary wage money payment intern",
        "answer": "**Payment depends entirely on the company — it is not mandatory.**\n\n- Some companies pay interns a daily or monthly wage\n- Some companies provide no payment at all\n- Whether you are paid or unpaid does **not** affect the validity or approval of your internship"
    },
    {
        "question": "What companies are eligible for internship? Which companies can I intern at? Company requirements",
        "answer": "**Eligible companies for IE internships:**\n\n- Any manufacturing or service company related to Industrial Engineering\n- Must have **at least 10 employees**\n- Must have **at least one engineer** on staff who can supervise the intern\n- Suitable sectors: manufacturing plants, banks, hospitals, logistics, consulting, IT, construction\n- Both domestic (Turkey) and international companies are eligible"
    },
    {
        "question": "How should I fill the internship logbook? Logbook daily report guidelines how to write",
        "answer": "**Internship logbook guidelines:**\n\n- Fill it out **every single working day
