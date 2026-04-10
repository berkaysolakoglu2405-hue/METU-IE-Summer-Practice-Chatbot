import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
 
# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="METU IE Summer Practice Chatbot",
    page_icon="🎓",
    layout="wide",
)
 
# ══════════════════════════════════════════════════════════════
#  API KEY — Streamlit Secrets'tan otomatik okunur
# ══════════════════════════════════════════════════════════════
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    API_KEY = os.environ.get("GOOGLE_API_KEY", "")
 
if not API_KEY:
    st.error(
        "🔑 **API Key not found.**\n\n"
        "Go to Streamlit Cloud → your app → ⋮ → Settings → Secrets and add:\n\n"
        "```toml\nGOOGLE_API_KEY = \"AIza...\"\n```"
    )
    st.stop()
 
# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f5f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #7a0000 0%, #4a0000 100%);
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 4px !important;
    padding: 6px 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
}
.main-header {
    background: linear-gradient(90deg, #8B0000, #CC0000);
    padding: 1.4rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(139,0,0,0.25);
}
.main-header h2 { color: white; margin: 0; font-size: 1.6rem; font-weight: 600; }
.main-header p  { color: #ffcccc; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
.footer {
    text-align: center; font-size: 0.75rem; color: #aaa;
    margin-top: 2rem; padding-top: 1rem;
    border-top: 1px solid #e0d8d0;
}
</style>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = """
=== METU IE SUMMER PRACTICE - OFFICIAL INFORMATION ===
 
GENERAL OVERVIEW:
METU Industrial Engineering Summer Practice program allows students to apply theoretical knowledge in real-world settings.
Two mandatory internships: IE 300 (after 3rd year) and IE 400 (after 4th year).
 
DURATION:
Both IE 300 and IE 400 require minimum 20 working days (4 weeks).
Weekends and official holidays do not count toward this requirement.
Any leave days given by the company are also deducted from the total.
 
APPLICATION PROCESS:
1. Fill online application at sp-ie.metu.edu.tr
2. Enter company details and internship dates
3. Upload required documents
4. Wait for department approval
5. Complete SGK (Social Security) registration
 
REQUIRED DOCUMENTS:
Before internship:
- Internship Application Form (online)
- SGK Form (for social security)
- Company Acceptance Letter
 
After internship:
- Internship Logbook (daily records)
- Evaluation Form (signed by supervisor)
- Internship Completion Report
 
IE 300 INTERNSHIP:
- For students completing their 3rd year
- Prerequisite: successful completion of IE 200
- Focus on basic engineering applications
- Manufacturing or service sector companies are eligible
- Emphasis on observation and hands-on experience
- Minimum 20 working days required
 
IE 400 INTERNSHIP:
- For students completing their 4th year
- Prerequisite: successful completion of IE 300 internship
- Focus on advanced engineering problems
- Covers engineering analysis, system design, or process improvement
- Minimum 20 working days required
- IE 300 and IE 400 cannot be done in the same summer
 
LOGBOOK:
- Must be filled out every working day
- Describe daily activities, observations, and lessons learned
- Must be signed/approved by company supervisor (daily or weekly)
- Submitted to the department at the end of the internship
 
INTERNSHIP REPORT:
- Can be written in English or Turkish
- Typical length: 20 to 40 pages
- Must include: company introduction, work carried out, skills gained, conclusion
- Submitted after the internship ends
 
EVALUATION:
- Based on the logbook quality and completeness
- Quality of the internship report
- Company supervisor evaluation form
- An oral presentation may be required
 
ELIGIBLE COMPANIES:
- Any manufacturing or service company related to Industrial Engineering
- Must have at least 10 employees
- Must have an engineer available to supervise the intern
- Suitable sectors: manufacturing plants, banks, hospitals, logistics companies, consulting firms
 
INTERNSHIP ABROAD:
- Allowed, but prior departmental approval is mandatory
- Same requirements apply (duration, documents, report)
- Erasmus+ internship grant opportunities available through the International Relations Office
 
SGK INSURANCE:
- Arranged by the department secretary before the internship starts
- Students are covered by occupational accident and occupational disease insurance
- Student must submit all required documents before the internship begins
 
PAYMENT:
- Internship payment is not mandatory; it depends on the company
- Some companies pay interns a daily or monthly wage
- Being paid or unpaid does not affect the validity of the internship
 
DEADLINES:
- Application deadlines are announced each semester by the department
- Summer internship applications are typically due in May or June
- Apply at least 3 weeks before the internship start date
- Check sp-ie.metu.edu.tr for current deadlines
 
WHAT IF INTERNSHIP IS NOT APPROVED:
- Student will be asked for additional information or documents
- After corrections, it will be re-evaluated
- In serious cases, the student may need to redo the internship
 
CONTACT AND OFFICIAL SOURCE:
- Official website: https://sp-ie.metu.edu.tr/en
- Department secretary office for in-person assistance
"""
 
FAQ_TEXT = """
Q: What documents are required for IE 300?
A: Before: Internship Application Form, Company Acceptance Letter, SGK form.
   After: Internship logbook, supervisor evaluation form, completion report.
 
Q: Can I do IE 300 and IE 400 in the same summer?
A: No. IE 300 must be successfully completed before starting IE 400. They must be done in separate summers.
 
Q: Can I intern abroad?
A: Yes, but prior departmental approval is mandatory. The same duration and document requirements apply.
 
Q: How should I fill the logbook?
A: Fill it every working day. Describe activities and observations. Have your supervisor sign it daily or weekly.
 
Q: Who arranges the SGK insurance?
A: The department secretary arranges it. Submit all required documents before the internship starts.
 
Q: Will I get paid during the internship?
A: It depends on the company. Payment is not required but some companies do pay interns.
 
Q: How long is the internship?
A: Minimum 20 working days (4 weeks) for both IE 300 and IE 400. Weekends and holidays do not count.
 
Q: When is the application deadline?
A: For summer internships, typically May-June. Check sp-ie.metu.edu.tr for exact current dates.
 
Q: What companies are eligible?
A: Any IE-related company with at least 10 employees and a supervising engineer on staff.
 
Q: How is the internship evaluated?
A: Logbook completeness, report quality, supervisor evaluation form, and possibly an oral presentation.
 
Q: What is the difference between IE 300 and IE 400?
A: IE 300 is for 3rd-year students focusing on basic engineering applications. IE 400 is for 4th-year students and focuses on advanced engineering problems. IE 300 must be completed before IE 400.
 
Q: Do I need IE 200 before IE 300?
A: Yes. Successful completion of IE 200 is a prerequisite for IE 300.
 
Q: Can my internship report be in Turkish?
A: Yes. The report can be written in either English or Turkish.
 
Q: What happens if my internship is rejected?
A: You will be asked for additional documents. After corrections it will be re-evaluated. In serious cases you may need to redo the internship.
 
Q: How do I start the application?
A: Go to sp-ie.metu.edu.tr, fill the online form with company details and dates, upload documents, and wait for department approval.
"""
 
# ══════════════════════════════════════════════════════════════
#  OUT-OF-SCOPE DETECTION
# ══════════════════════════════════════════════════════════════
SCOPE_KEYWORDS = [
    "internship", "training", "ie 300", "ie 400", "summer practice",
    "staj", "application", "apply", "document", "form", "report",
    "evaluation", "deadline", "date", "sgk", "insurance", "company",
    "acceptance", "approval", "metu", "industrial engineering",
    "summer", "sp-ie", "intern", "salary", "wage", "logbook",
    "daily", "abroad", "erasmus", "requirement", "secretary",
    "department", "duration", "working days", "weeks", "grade",
    "complete", "supervisor", "signed", "submit", "ie200", "ie 200",
    "prerequisite", "reject", "report", "turkish", "english",
]
 
def is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return not any(kw in q for kw in SCOPE_KEYWORDS)
 
# ══════════════════════════════════════════════════════════════
#  FAISS VECTOR STORE (runs once, cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    full_text = KNOWLEDGE_BASE + "\n\n" + FAQ_TEXT
    if os.path.exists("scraped_data.txt"):
        with open("scraped_data.txt", "r", encoding="utf-8") as f:
            full_text += "\n\n" + f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embeddings)
 
# ══════════════════════════════════════════════════════════════
#  GEMINI ANSWER (via LangChain — no google.generativeai import)
# ══════════════════════════════════════════════════════════════
def ask_gemini(question: str, context: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=API_KEY,
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant for METU Industrial Engineering Summer Practice "
            "(IE 300 & IE 400). Answer using ONLY the context provided. "
            "If the answer is not in the context, say: 'For the most accurate information, "
            "please visit sp-ie.metu.edu.tr or contact the IE Department secretary.' "
            "Use bullet points for lists. Be concise and professional."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content
 
# ══════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h2>🎓 METU IE Summer Practice Chatbot</h2>
    <p>IE 300 &amp; IE 400 · Applications · Documents · Deadlines · Process</p>
</div>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════
#  UI — SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 METU IE\nSummer Practice")
    st.markdown("---")
    st.markdown("### 📚 Sample Questions")
    st.caption("Click to ask instantly:")
 
    samples = [
        "What documents are required for IE 300?",
        "How do I apply for the internship?",
        "What is the difference between IE 300 and IE 400?",
        "Can I do IE 300 and IE 400 in the same summer?",
        "Can I do my internship abroad?",
        "How should I fill the logbook?",
        "Who arranges the SGK insurance?",
        "Will I be paid during my internship?",
        "How long does the internship last?",
        "What happens if my internship is rejected?",
    ]
    for q in samples:
        if st.button(q, key=f"btn_{q[:25]}"):
            st.session_state["prefill"] = q
 
    st.markdown("---")
    st.markdown("📌 **Official source:**")
    st.markdown("[sp-ie.metu.edu.tr/en](https://sp-ie.metu.edu.tr/en)")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()
 
# ══════════════════════════════════════════════════════════════
#  UI — CHAT
# ══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm the **METU IE Summer Practice Assistant**.\n\n"
                "I can answer your questions about **IE 300** and **IE 400** internships:\n"
                "- 📋 How to apply\n"
                "- 📄 Required documents\n"
                "- 📅 Deadlines\n"
                "- 📝 Logbook & report preparation\n"
                "- 🏢 Eligible companies\n"
                "- 🌍 Internships abroad\n\n"
                "What would you like to know?"
            ),
        }
    ]
 
prefill = st.session_state.pop("prefill", None)
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
user_input = st.chat_input("Ask anything about METU IE Summer Practice…") or prefill
 
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        if is_out_of_scope(user_input):
            answer = (
                "🚫 **This question is outside the scope of METU IE Summer Practice.**\n\n"
                "I can only help with IE 300 and IE 400 internship questions — "
                "application process, required documents, logbook, report, "
                "deadlines, SGK insurance, and eligible companies.\n\n"
                "Please ask about one of those topics! 🎓"
            )
            st.markdown(answer)
        else:
            with st.spinner("Thinking…"):
                try:
                    vector_store = build_vector_store()
                    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                    retrieved_docs = retriever.invoke(user_input)
                    context = "\n\n".join([d.page_content for d in retrieved_docs])
                    answer = ask_gemini(user_input, context)
                    st.markdown(answer)
                except Exception as e:
                    answer = (
                        f"⚠️ **An error occurred:** `{e}`\n\n"
                        "Please check that your API key is correct in Streamlit Secrets."
                    )
                    st.markdown(answer)
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
 
st.markdown(
    '<div class="footer">'
    'METU IE Summer Practice Assistant · Powered by Google Gemini · '
    '<a href="https://sp-ie.metu.edu.tr/en" target="_blank">sp-ie.metu.edu.tr</a>'
    '</div>',
    unsafe_allow_html=True,
)
