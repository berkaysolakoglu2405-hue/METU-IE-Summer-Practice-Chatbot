import streamlit as st
import google.generativeai as genai
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="METU IE Summer Practice Chatbot", page_icon="🎓")

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

# ── Knowledge Base ─────────────────────────────────────────────
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
- Suitable sectors include: manufacturing plants, banks, hospitals, logistics companies, consulting firms

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
A: You need the Internship Application Form, Company Acceptance Letter, and SGK form before starting.
After the internship: logbook, supervisor evaluation form, and completion report.

Q: Can I do IE 300 and IE 400 in the same summer?
A: No. IE 300 must be successfully completed before starting IE 400.

Q: Can I intern abroad?
A: Yes, but you must get departmental approval beforehand. The same requirements apply.

Q: How should I fill the logbook?
A: Fill it every working day. Write your activities and observations. Have your supervisor sign it daily or weekly.

Q: Who arranges the SGK insurance?
A: The department secretary arranges it. Submit your documents before the internship starts.

Q: Will I get paid during the internship?
A: It depends on the company. Payment is not required but some companies do pay interns.

Q: How long is the internship?
A: Minimum 20 working days (4 weeks) for both IE 300 and IE 400. Weekends and holidays do not count.

Q: When is the application deadline?
A: Deadlines are announced each semester. For summer internships, typically May-June. Check sp-ie.metu.edu.tr.

Q: What companies are eligible?
A: Any company related to Industrial Engineering with at least 10 employees and a supervising engineer.

Q: How is the internship evaluated?
A: Based on the logbook, report quality, supervisor evaluation form, and possibly an oral presentation.
"""

# ── Out-of-scope detection ─────────────────────────────────────
SCOPE_KEYWORDS = [
    "internship", "training", "ie 300", "ie 400", "summer practice",
    "staj", "application", "apply", "document", "form", "report",
    "evaluation", "deadline", "date", "sgk", "insurance", "company",
    "acceptance", "approval", "metu", "industrial engineering",
    "summer", "sp-ie", "intern", "salary", "wage", "logbook",
    "daily", "abroad", "erasmus", "requirement", "secretary",
    "department", "duration", "working days", "weeks", "grade",
    "complete", "supervisor", "signed", "submit",
]

def is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return not any(kw in q for kw in SCOPE_KEYWORDS)

# ── Build FAISS vector store (cached so it only runs once) ─────
@st.cache_resource(show_spinner=False)
def build_vector_store(api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    full_text = KNOWLEDGE_BASE + "\n\n" + FAQ_TEXT

    if os.path.exists("scraped_data.txt"):
        with open("scraped_data.txt", "r", encoding="utf-8") as f:
            full_text += "\n\n" + f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=c) for c in chunks]

    return FAISS.from_documents(docs, embeddings)

# ── Ask Gemini with retrieved context ─────────────────────────
def ask_gemini(api_key: str, question: str, context: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""You are a helpful assistant for METU Industrial Engineering Summer Practice (IE 300 & IE 400).
Answer the student's question using ONLY the context below.
If the answer is not in the context, say: "For the most accurate information, please visit sp-ie.metu.edu.tr or contact the department secretary."
Use bullet points for lists. Be concise and professional.

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
    <p>IE 300 & IE 400 · Applications · Documents · Deadlines · Process</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at aistudio.google.com/app/apikey",
    )
    if api_key:
        st.success("✅ API Key set")
    else:
        st.info("🔑 Enter your API Key to start.")

    st.markdown("---")
    st.markdown("### 📚 Sample Questions")
    samples = [
        "What documents are required for IE 300?",
        "How do I apply for the internship?",
        "Can I do my internship abroad?",
        "Can I do IE 300 and IE 400 in the same summer?",
        "How should I fill the logbook?",
        "Who arranges the SGK insurance?",
        "Will I be paid during my internship?",
        "How long does the internship last?",
    ]
    for q in samples:
        if st.button(q, key=q):
            st.session_state["prefill"] = q

    st.markdown("---")
    st.markdown("📌 **Source:** [sp-ie.metu.edu.tr](https://sp-ie.metu.edu.tr/en)")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Stop if no API key
if not api_key:
    st.info("👈 Please enter your **Google Gemini API Key** in the sidebar to start chatting.")
    st.stop()

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your METU IE Summer Practice assistant.\n\n"
                "I can answer questions about **IE 300** and **IE 400** internships — "
                "applications, documents, deadlines, logbooks, reports, and more.\n\n"
                "What would you like to know?"
            ),
        }
    ]

# Prefill from sidebar button
prefill = st.session_state.pop("prefill", None)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about METU IE Summer Practice…") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_out_of_scope(user_input):
            answer = (
                "🚫 **This question is outside the scope of METU IE Summer Practice.**\n\n"
                "I can only help with questions about IE 300 and IE 400 internships, "
                "such as the application process, required documents, logbook, report, "
                "deadlines, and SGK insurance. Please ask about one of these topics! 🎓"
            )
            st.markdown(answer)
        else:
            with st.spinner("Thinking…"):
                try:
                    vector_store = build_vector_store(api_key)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(user_input)
                    context = "\n\n".join([d.page_content for d in docs])
                    answer = ask_gemini(api_key, user_input, context)
                    st.markdown(answer)
                except Exception as e:
                    answer = f"⚠️ Error: `{e}`\n\nPlease check your API key and try again."
                    st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<div class="footer">METU IE Summer Practice Assistant · '
    'Powered by Google Gemini · Source: sp-ie.metu.edu.tr</div>',
    unsafe_allow_html=True,
)
