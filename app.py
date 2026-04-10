import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
#  KNOWLEDGE BASE  (FAQ answers built-in — no API needed)
# ══════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = """
QUESTION: What documents are required for IE 300?
ANSWER: You need the following documents:
Before the internship:
- Internship Application Form (filled online at sp-ie.metu.edu.tr)
- Company Acceptance Letter (signed by company)
- SGK Form (for social security registration, arranged by department)
After the internship:
- Internship Logbook (daily records signed by supervisor)
- Supervisor Evaluation Form (filled and signed by company supervisor)
- Internship Completion Report (20-40 pages)

QUESTION: What documents are required for IE 400?
ANSWER: The same documents as IE 300 are required for IE 400:
Before: Application Form, Company Acceptance Letter, SGK Form.
After: Logbook, Supervisor Evaluation Form, Completion Report.
Additionally, you must have already completed IE 300 before starting IE 400.

QUESTION: What are the requirements for IE 300?
ANSWER: IE 300 (Industrial Training I) requirements:
- You must have successfully completed IE 200 before applying
- Minimum duration: 20 working days (4 weeks)
- Must be done at a manufacturing or service sector company
- The company must have at least 10 employees and a supervising engineer
- You must get departmental approval before starting
- Focus is on basic engineering applications and observation

QUESTION: What are the requirements for IE 400?
ANSWER: IE 400 (Industrial Training II) requirements:
- You must have successfully completed IE 300 first
- Minimum duration: 20 working days (4 weeks)
- Focus is on advanced engineering problems: system design, process improvement, engineering analysis
- Same company eligibility rules as IE 300 apply
- IE 300 and IE 400 cannot be done in the same summer

QUESTION: How do I apply for the internship?
ANSWER: To apply for the internship, follow these steps:
1. Go to sp-ie.metu.edu.tr and fill the online application form
2. Enter your company details and planned internship dates
3. Upload the required documents (acceptance letter, etc.)
4. Submit and wait for department approval
5. After approval, the department will handle SGK (social security) registration
Important: Apply at least 3 weeks before your internship start date.

QUESTION: How long does the internship last?
ANSWER: Both IE 300 and IE 400 require a minimum of 20 working days (approximately 4 calendar weeks).
- Weekends and official public holidays do NOT count toward this 20-day requirement
- Any leave days given by the company are also deducted from the total
- You must complete the full 20 working days for the internship to be valid

QUESTION: Can I do IE 300 and IE 400 in the same summer?
ANSWER: No. IE 300 and IE 400 cannot be done in the same summer or simultaneously.
IE 300 must be fully completed and approved before you can begin IE 400.
They must be completed in separate summers (or separate terms).

QUESTION: What is the difference between IE 300 and IE 400?
ANSWER: IE 300 vs IE 400:
- IE 300 is for students who have completed their 3rd year. It focuses on basic engineering applications, observation, and hands-on experience in a manufacturing or service company.
- IE 400 is for students who have completed their 4th year. It focuses on advanced engineering problems such as system design, process improvement, and engineering analysis.
- IE 300 must be completed before starting IE 400.
- Both require minimum 20 working days.

QUESTION: Can I do my internship abroad?
ANSWER: Yes, internships abroad are allowed. However:
- You must get prior approval from the IE Department before starting
- The same requirements apply: minimum 20 working days, same documents, same report
- For Erasmus+ internship grant funding, contact the METU International Relations Office
- Apply well in advance as international approvals may take longer

QUESTION: Who arranges the SGK insurance?
ANSWER: The SGK (Social Security Institution) insurance registration is arranged by the IE Department secretary — you do not need to do it yourself.
- Submit all your required documents to the department before your internship starts
- Students are covered by occupational accident and occupational disease insurance during the internship
- Do not start your internship before the SGK registration is confirmed

QUESTION: Will I be paid during the internship?
ANSWER: Payment during the internship depends entirely on the company — it is not mandatory.
- Some companies pay interns a daily or monthly wage
- Some companies provide no payment
- Whether you are paid or unpaid does NOT affect the validity or approval of your internship

QUESTION: What companies are eligible for internship?
ANSWER: Eligible companies for IE internships:
- Any manufacturing or service company related to Industrial Engineering activities
- Must have at least 10 employees
- Must have at least one engineer on staff who can supervise the intern
- Suitable sectors include: manufacturing plants, banks, hospitals, logistics companies, consulting firms, IT companies, construction firms
- Both domestic (Turkey) and international companies are eligible (with prior approval for abroad)

QUESTION: How should I fill the internship logbook?
ANSWER: Internship logbook guidelines:
- Fill it out every single working day — do not skip days
- For each day, describe: what tasks you did, what you observed, what you learned
- Write clearly and in enough detail to show your engineering activities
- Your company supervisor must sign/approve it (daily or at least weekly)
- Submit the completed and signed logbook to the department at the end of your internship

QUESTION: How should I prepare the internship report?
ANSWER: Internship report guidelines:
- Language: can be written in English or Turkish
- Length: typically 20 to 40 pages
- Required sections:
  1. Company introduction and organizational structure
  2. Description of work carried out during internship
  3. Engineering skills and knowledge gained
  4. Conclusion and personal evaluation
- Submit the report to the department after your internship ends

QUESTION: How is the internship evaluated?
ANSWER: The internship is evaluated based on:
- Completeness and quality of the internship logbook
- Quality and content of the internship report
- Supervisor evaluation form (filled and signed by your company supervisor)
- In some cases, an oral presentation or interview with the department may be required
All components must be satisfactory for the internship to be approved.

QUESTION: What happens if my internship is not approved?
ANSWER: If the department finds your internship incomplete or insufficient:
- You will be notified and asked to provide additional information or documents
- After you make the required corrections, it will be re-evaluated
- In cases of serious deficiency, you may be required to redo the internship
- It is therefore important to keep a detailed logbook and write a thorough report

QUESTION: When is the application deadline?
ANSWER: Application deadlines are announced by the IE Department each semester.
- For summer internships: applications are typically due in May or June
- You should apply at least 3 weeks before your planned internship start date
- Always check sp-ie.metu.edu.tr or the department bulletin board for the exact current deadlines
- Missing the deadline may mean you cannot do your internship that summer

QUESTION: Where can I find official forms and information?
ANSWER: All official internship forms, announcements, guidelines, and deadlines are available at:
- Website: https://sp-ie.metu.edu.tr/en
- You can also visit the IE Department secretary's office in person for help with forms and paperwork
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
    "paid", "pay", "company", "eligible", "how long", "difference",
]

def is_out_of_scope(query: str) -> bool:
    return not any(kw in query.lower() for kw in SCOPE_KEYWORDS)

# ══════════════════════════════════════════════════════════════
#  FAISS VECTOR STORE — HuggingFace (no API key needed at all)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading knowledge base… (first load takes ~30 seconds)")
def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\nQUESTION:", "\n\n", "\n"]
    )
    chunks = splitter.split_text(KNOWLEDGE_BASE)
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embeddings)

def get_answer(question: str) -> str:
    """Retrieve the most relevant chunks and return them as the answer."""
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved = retriever.invoke(question)
    if not retrieved:
        return (
            "I could not find specific information about that. "
            "Please visit [sp-ie.metu.edu.tr/en](https://sp-ie.metu.edu.tr/en) "
            "or contact the IE Department secretary."
        )
    # Return the best matching chunk(s), cleaned up
    results = []
    for doc in retrieved:
        text = doc.page_content.strip()
        if text and text not in results:
            results.append(text)
    return "\n\n---\n\n".join(results[:2])

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
        "What happens if my internship is not approved?",
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
        else:
            with st.spinner("Searching knowledge base…"):
                answer = get_answer(user_input)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<div class="footer">'
    'METU IE Summer Practice Assistant · '
    'Source: <a href="https://sp-ie.metu.edu.tr/en" target="_blank">sp-ie.metu.edu.tr</a>'
    '</div>',
    unsafe_allow_html=True,
)
