import streamlit as st
from rag_pipeline import MedicalRAG
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("img.jpg")


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PubMed Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Light overlay for readability */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(255,255,255,0.9);
            z-index: -1;
        }}

        html, body, [class*="css"] {{
            color: black !important;
        }}

        h1, h2, h3, h4 {{
            color: #97d8f3 !important;
        }}

        .stTextInput>div>div>input {{
            background-color: white;
            color: black !important;
            border-radius: 8px;
        }}

        .stButton>button {{
            background-color: #1976d2;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            height: 3em;
        }}

        .stButton>button:hover {{
            background-color: #0d47a1;
        }}
    </style>
""", unsafe_allow_html=True)



st.markdown("""
# 🧠 PubMed Research Assistant  
### AI-Powered Medical Research Q&A System  
Ask research-level questions and get answers grounded in PubMed abstracts.
""")


# ---------------- INPUT SECTION ----------------
with st.container():
    st.markdown("## 🔎 Load Research Topic")

    query_topic = st.text_input(
        "Enter a medical topic:",
        "diabetes treatment"
    )

    if st.button("🚀 Load Research Papers"):
        with st.spinner("Fetching and indexing research papers..."):
            rag = MedicalRAG()
            rag.load_data(query_topic)
            st.session_state.rag = rag
        st.success("Research papers loaded successfully!")

st.markdown("---")

# ---------------- Q&A SECTION ----------------
if "rag" in st.session_state:

    st.markdown("## 💬 Ask Your Question")

    question = st.text_input("Enter your research question:")

    if st.button("🧪 Generate Answer"):
        with st.spinner("Analyzing research papers..."):
            answer, citations = st.session_state.rag.ask(question)

        st.markdown("### 📌 Answer")
        st.info(answer)

        st.markdown("### 📚 Citations (Source Context)")
        for i, c in enumerate(citations):
            st.markdown(f"**Paper {i+1}:**")
            st.success(c)

st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <center>
    Built with ❤️ using RAG, FAISS, Sentence Transformers & PubMed API  
    </center>
    """,
    unsafe_allow_html=True
)
