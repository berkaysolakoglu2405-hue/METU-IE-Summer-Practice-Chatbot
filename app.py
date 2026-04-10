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
        "answer": "**Internship logbook guidelines:**\n\n- Fill it out **every single working day** — do not skip days\n- For each day, write: what tasks you did, what you observed, what you learned\n- Write clearly and in enough detail to show your engineering activities\n- Your company supervisor must **sign/approve** it (daily or at least weekly)\n- Submit the completed, signed logbook to the department at the end of your internship"
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
        "answer": "**Application deadlines:**\n\n- Deadlines are announced by the IE Department each semester\n- For summer internships: applications are typically due in **May or June**\n- Apply at least **3 weeks before** your planned internship start date\n- Check **sp-ie.metu.edu.tr** or the department bulletin board for the exact current deadlines\n- Missing the deadline may mean you cannot do your internship that summer"
    },
    {
        "question": "Where can I find official forms information announcements? Forms download official website",
        "answer": "**Official sources:**\n\n- 🌐 Website: **https://sp-ie.metu.edu.tr/en**\n- 🏢 IE Department secretary's office (in person)\n\nAll internship forms, announcements, guidelines, and deadlines are published there."
    },
    {
        "question": "How can I find a summer practice? How to find a company for internship? Find internship place",
        "answer": "**How to find a company for your summer practice:**\n\n- Start searching **early** — at least 2-3 months before the internship period\n- Look for companies in IE-related sectors: manufacturing, logistics, banking, hospitals, consulting, IT\n- The company must have **at least 10 employees** and an engineer who can supervise you\n- Check METU career fairs, LinkedIn, company websites, and your personal network\n- Once a company agrees, get their **Acceptance Letter** and apply through sp-ie.metu.edu.tr\n- You can intern **abroad** too — just get departmental approval first"
    }
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
#  KİMLİĞİ GİZLENMİŞ, BARAJI KALDIRILMIŞ MELEZ BEYİN (KOTA KORUMALI)
# ══════════════════════════════════════════════════════════════
def ask_gemini(user_question: str) -> str:
    vector_store = build_vector_store()
    
    # En iyi 3 eşleşmeyi koşulsuz şartsız çekiyoruz.
    results = vector_store.similarity_search(user_question, k=3)
    context = "\n\n---\n\n".join([doc.metadata["answer"] for doc in results])

    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not available_models:
            return "⚠️ HATA: API Key yetkisiz veya modellere erişilemiyor."
            
        # SADECE BU KISMI DEĞİŞTİRDİM (2.0 ve 2.5 yasak)
        safe_models = [m for m in available_models if "2.0" not in m and "2.5" not in m and "vision" not in m]
        
        if not safe_models:
            return "⚠️ HATA: Geçerli bedava model bulunamadı."
            
        chosen_model = safe_models[0]
        for m in safe_models:
            if "1.5-flash" in m:
                chosen_model = m
                break
                
        clean_model_name = chosen_model.replace("models/", "")
        model = genai.GenerativeModel(clean_model_name)
        
    except Exception as e:
        return f"API bağlantı hatası: {e}"
    
    prompt = f"""You are the official 'METU IE Summer Practice Assistant'.
    
    CRITICAL IDENTITY RULE: 
    NEVER mention that you are an AI, a language model, Gemini, or developed by Google. Act like a helpful human-like assistant dedicated to METU IE students. If someone asks "who are you" or "are you an AI", purely introduce yourself as the METU IE Summer Practice Assistant.
    
    Here is the retrieved information from the official internship database:
    {context}
    
    Task Guidelines:
    1. INTERNSHIP QUESTIONS: If the user asks about METU IE Internships, answer accurately using ONLY the information provided in the context above.
    2. OUT OF SCOPE QUESTIONS: If the user asks something completely unrelated to internships (e.g., math, coding, history, general chat), answer perfectly using your general knowledge, but ALWAYS STAY IN CHARACTER as the METU IE Assistant. Do not break character.

    User Question: {user_question}
    Answer:"""

    response = model.generate_content(prompt)
    return response.text

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

st.markdown(
    '<div class="footer">'
    'METU IE Summer Practice Assistant · '
    'Source: <a href="https://sp-ie.metu.edu.tr/en" target="_blank">sp-ie.metu.edu.tr</a>'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
#  AUTO-SCROLL (AŞAĞI IŞINLANMA) HİLESİ
# ══════════════════════════════════════════════════════════════
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
