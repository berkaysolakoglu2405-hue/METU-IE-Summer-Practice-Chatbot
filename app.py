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
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] { 
    font-family: 'Inter', sans-serif; 
}
[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #7a0000 0%, #4a0000 100%) !important; 
}
[data-testid="stSidebar"] * { 
    color: #ffffff !important; 
}
[data-testid="stSidebar"] .stButton > button { 
    background: rgba(255,255,255,0.12) !important; 
    color: #ffffff !important; 
    border: 1px solid rgba(255,255,255,0.25) !important; 
    border-radius: 8px !important; 
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
.main-header h2 { color: white !important; margin: 0; font-size: 1.6rem; font-weight: 600; }
.main-header p  { color: #ffcccc !important; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
.footer { 
    text-align: center; 
    font-size: 0.75rem; 
    color: #888; 
    margin-top: 2rem; 
    padding-top: 1rem; 
    border-top: 1px solid rgba(128,128,128,0.2); 
}
[data-testid="stChatMessage"] {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  Q&A DATABASE
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
        "question": "What are the requirements for IE 300? IE 300 requirements prerequisites conditions sector company",
        "answer": "**IE 300 (Industrial Training I) requirements:**\n\n- You must have successfully completed **IE 200** before applying\n- Minimum duration: **20 working days (4 weeks)**\n- Must be done at a **manufacturing (production) company only** — service sector is NOT eligible\n- The company must have a supervising engineer on staff\n- You must get departmental approval before starting\n- Focus is on production processes and hands-on manufacturing experience"
    },
    {
        "question": "What are the requirements for IE 400? IE 400 requirements prerequisites conditions sector company",
        "answer": "**IE 400 (Industrial Training II) requirements:**\n\n- You must have successfully completed **IE 300** before applying\n- Minimum duration: **20 working days (4 weeks)**\n- Can be done at **manufacturing OR service sector companies** (banks, hospitals, logistics, IT, consulting, etc.)\n- Focus is on advanced engineering: system design, process improvement, engineering analysis\n- IE 300 and IE 400 **cannot** be done in the same summer"
    },
    {
        "question": "Can I do IE 300 at a service company bank hospital logistics IT? Which sector is eligible for IE 300?",
        "answer": "**No. IE 300 must be done at a manufacturing (production) company only.**\n\n- Service sector companies such as banks, hospitals, IT firms, logistics companies, and consulting firms are **NOT eligible** for IE 300\n- IE 300 specifically focuses on production and manufacturing processes\n- Service sector companies are only eligible for **IE 400**"
    },
    {
        "question": "Can I do IE 400 at a service company bank hospital logistics IT? Which sector is eligible for IE 400?",
        "answer": "**Yes. IE 400 can be done at both manufacturing and service sector companies.**\n\nEligible sectors for IE 400:\n- Manufacturing and production facilities\n- Banks and financial institutions\n- Hospitals and healthcare organizations\n- Logistics and supply chain companies\n- IT and software companies\n- Consulting firms\n\nThe company must have a supervising engineer related to Industrial Engineering."
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
        "answer": "**IE 300 vs IE 400:**\n\n| | IE 300 | IE 400 |\n|---|---|---|\n| Who | 3rd-year students | 4th-year students |\n| Prerequisite | IE 200 completed | IE 300 completed |\n| Eligible sector | **Manufacturing only** | Manufacturing + Service |\n| Focus | Production, observation | Advanced engineering, system design |\n| Duration | Min. 20 working days | Min. 20 working days |"
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
        "answer": "**Eligible companies:**\n\n- **IE 300:** Manufacturing (production) companies only\n- **IE 400:** Both manufacturing and service sector companies\n- Must have a supervising engineer on staff\n- Both domestic (Turkey) and international companies are eligible"
    },
    {
        "question": "How should I fill the internship logbook? Logbook daily report guidelines how to write",
        "answer": "**Internship logbook guidelines:**\n\n- Fill it out **every single working day** — do not skip days\n- For each day, write: what tasks you did, what you observed, what you learned\n- Your company supervisor must **sign/approve** it (daily or at least weekly)\n- Submit the completed, signed logbook to the department at the end of your internship"
    },
    {
        "question": "How should I prepare the internship report? Report format length structure writing",
        "answer": "**Internship report guidelines:**\n\n- **Language:** English or Turkish — your choice\n- **Length:** typically 20 to 40 pages\n- **Required sections:**\n  1. Company introduction and organisational structure\n  2. Detailed description of work carried out\n  3. Engineering skills and knowledge gained\n  4. Conclusion and personal evaluation\n- Submit the report to the department after your internship ends"
    },
    {
        "question": "How is the internship evaluated? Grading assessment criteria pass fail",
        "answer": "**Internship evaluation criteria:**\n\n- Completeness and quality of the **internship logbook**\n- Quality and content of the **internship report**\n- **Supervisor evaluation form** (filled and signed by your company supervisor)\n- In some cases, an **oral presentation** with the department may be required\n\nAll components must be satisfactory for the internship to be approved."
    },
    {
        "question": "What happens if my internship is not approved rejected? Fail internship rejected incomplete",
        "answer": "**If your internship is not approved:**\n\n- You will be notified and asked to provide **additional information or documents**\n- After you make the required corrections, it will be **re-evaluated**\n- In cases of serious deficiency, you may be required to **redo the internship**\n\nTo avoid rejection: keep a detailed daily logbook and write a thorough, well-structured report."
    },
    {
        "question": "When is the application deadline? Last date to apply summer practice deadline",
        "answer": "**Application deadlines:**\n\n- Deadlines are announced by the IE Department each semester\n- For summer internships: applications are typically due in **May or June**\n- Apply at least **3 weeks before** your planned internship start date\n- Check **sp-ie.metu.edu.tr** or the department bulletin board for the exact current deadlines"
    },
    {
        "question": "Where can I find official forms information announcements? Forms download official website",
        "answer": "**Official sources:**\n\n- 🌐 Website: **https://sp-ie.metu.edu.tr/en**\n- 🏢 IE Department secretary's office (in person)\n\nAll internship forms, announcements, guidelines, and deadlines are published there."
    },
    {
        "question": "How can I find a summer practice? How to find a company for internship? Find internship place",
        "answer": "**How to find a company for your summer practice:**\n\n- Start searching **early** — at least 2-3 months before the internship period\n- **IE 300:** manufacturing/production companies only\n- **IE 400:** manufacturing or service sector — both are fine\n- Check METU career fairs, LinkedIn, company websites, and your personal network\n- Once a company agrees, get their **Acceptance Letter** and apply through sp-ie.metu.edu.tr"
    }
]

# ══════════════════════════════════════════════════════════════
#  VECTOR STORE
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading knowledge base… (~30 seconds on first load)")
def build_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    docs = [
        Document(
            page_content=qa["question"] + " " + qa["answer"],
            metadata={"answer": qa["answer"], "idx": i}
        )
        for i, qa in enumerate(QA_DATABASE)
    ]
    return FAISS.from_documents(docs, embeddings)

# ══════════════════════════════════════════════════════════════
#  MODEL SEÇİMİ — 2.0 ve 2.5 atla, 1.5-flash'ı bul
# ══════════════════════════════════════════════════════════════
def ask_gemini(user_question: str) -> str:
    vector_store = build_vector_store()
    results = vector_store.similarity_search(user_question, k=3)
    context = "\n\n---\n\n".join([doc.metadata["answer"] for doc in results])

    # Model seçimi: 2.0/2.5 atla, 1.5-flash'ı bul
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
            and "vision" not in m.name
            and "embedding" not in m.name
            and "2.0" not in m.name
            and "2.5" not in m.name
        ]
        if not available_models:
            return "⚠️ No available model found. Please check your API key."

        chosen_model = available_models[0]
        for m in available_models:
            if "1.5-flash" in m:
                chosen_model = m
                break

        clean_model_name = chosen_model.replace("models/", "")
        model = genai.GenerativeModel(clean_model_name)

    except Exception as e:
        return f"API connection error: {e}"

    prompt = f"""You are the official METU IE Summer Practice Assistant for METU Industrial Engineering students.

You have access to the official internship database below (CONTEXT). Use it as your primary source.

RULES:
1. Read the CONTEXT carefully. Your answer MUST agree with what the context says — never contradict it.
2. Start your answer by directly stating what the context says (yes/no/how), then explain.
3. Do NOT add your own opinion or knowledge that contradicts the context.
4. If the answer is genuinely not in the context, say so briefly and suggest sp-ie.metu.edu.tr.
5. Never say you are an AI, Gemini, or made by Google.
6. Be concise and clear. Use bullet points when listing items.

CONTEXT FROM OFFICIAL DATABASE:
{context}

Student's question: {user_question}
Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Model error (`{clean_model_name}`): {e}"

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h2>🎓 METU IE Summer Practice Chatbot</h2>
    <p>IE 300 &amp; IE 400 · Applications · Documents · Deadlines · Process</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 METU IE\nSummer Practice")
    st.markdown("---")
    st.markdown("### 📚 Sample Questions")
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
        if st.button(q, key=f"btn_{q[:15]}"):
            st.session_state["prefill"] = q
    st.markdown("---")
    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm the **METU IE Summer Practice Assistant**.\n\n"
                "I can answer your questions about **IE 300** and **IE 400** internships, "
                "or we can chat about anything else you'd like! 🚀"
            )
        }
    ]

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

st.markdown(
    '<div class="footer">'
    'METU IE Summer Practice Assistant · '
    'Source: <a href="https://sp-ie.metu.edu.tr/en" target="_blank">sp-ie.metu.edu.tr</a>'
    '</div>',
    unsafe_allow_html=True,
)

components.html("""
<script>
    var chatHistory = window.parent.document.querySelector('.main');
    if (chatHistory) {
        chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    }
</script>
""", height=0)
