import os
import json
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

st.set_page_config(page_title="Resume RAG", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_split(uploaded_file):
    """Save an uploaded PDF to a temp file, load it, split into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)

    full_text = "\n".join(p.page_content for p in pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["source"] = uploaded_file.name

    return chunks, full_text


def build_vectorstore(all_chunks, embeddings):
    return FAISS.from_documents(all_chunks, embeddings)


MATCH_PROMPT = PromptTemplate(
    input_variables=["jd", "resume"],
    template="""You are a recruiting assistant. Compare the RESUME to the JOB DESCRIPTION.

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

Respond with ONLY a valid JSON object (no markdown, no code fences) in this exact shape:
{{
  "score": <integer 0-100, overall fit>,
  "strengths": ["short bullet", "short bullet"],
  "gaps": ["short bullet", "short bullet"]
}}""",
)


def score_resume(llm, jd_text, resume_text):
    prompt = MATCH_PROMPT.format(jd=jd_text[:6000], resume=resume_text[:6000])
    resp = llm.invoke(prompt)
    raw = resp.content.strip()
    raw = raw.strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "strengths": [], "gaps": [], "raw": raw}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "resumes" not in st.session_state:
    st.session_state.resumes = {}  # filename -> full_text
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------------------------------------------------------------------
# Sidebar: upload + process
# ---------------------------------------------------------------------------
st.sidebar.title("📄 Resume RAG")

if not GROQ_API_KEY:
    st.sidebar.error("GROQ_API_KEY not found. Add it to a .env file (see .env.example).")

uploaded_files = st.sidebar.file_uploader(
    "Upload resume PDFs", type=["pdf"], accept_multiple_files=True
)

if st.sidebar.button("Process Resumes", disabled=not uploaded_files):
    embeddings = get_embeddings()
    all_chunks = []
    with st.spinner("Reading and indexing resumes..."):
        for f in uploaded_files:
            chunks, full_text = load_and_split(f)
            st.session_state.resumes[f.name] = full_text
            all_chunks.extend(chunks)
        st.session_state.vectorstore = build_vectorstore(all_chunks, embeddings)
    st.sidebar.success(f"Indexed {len(uploaded_files)} resume(s).")

if st.session_state.resumes:
    st.sidebar.markdown("**Loaded resumes:**")
    for name in st.session_state.resumes:
        st.sidebar.write(f"- {name}")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_qa, tab_match = st.tabs(["💬 Ask Questions", "🎯 Match to Job Description"])

with tab_qa:
    st.header("Ask questions about the uploaded resumes")
    question = st.text_input("Your question", placeholder="e.g. Who has AWS experience?")
    k = st.slider("Number of chunks to retrieve", 2, 10, 4)

    if st.button("Get Answer", disabled=not question):
        if st.session_state.vectorstore is None:
            st.warning("Upload and process at least one resume first.")
        elif get_llm() is None:
            st.error("GROQ_API_KEY missing — set it in .env and restart.")
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
            qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                retriever=retriever,
                return_source_documents=True,
            )
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": question})
            st.markdown("### Answer")
            st.write(result["result"])
            with st.expander("Sources"):
                for doc in result["source_documents"]:
                    st.markdown(f"**{doc.metadata.get('source', 'unknown')}**")
                    st.caption(doc.page_content[:300] + "...")

with tab_match:
    st.header("Rank resumes against a job description")
    jd_text = st.text_area("Paste the job description here", height=220)

    if st.button("Rank Resumes", disabled=not jd_text):
        if not st.session_state.resumes:
            st.warning("Upload and process at least one resume first.")
        elif get_llm() is None:
            st.error("GROQ_API_KEY missing — set it in .env and restart.")
        else:
            llm = get_llm()
            results = []
            progress = st.progress(0.0)
            names = list(st.session_state.resumes.keys())
            for i, name in enumerate(names):
                score = score_resume(llm, jd_text, st.session_state.resumes[name])
                results.append({"file": name, **score})
                progress.progress((i + 1) / len(names))

            results.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))

            st.markdown("### Ranking")
            for r in results:
                score = r.get("score")
                label = f"{score}/100" if score is not None else "parse error"
                with st.expander(f"{r['file']} — {label}", expanded=True):
                    if score is None:
                        st.write(r.get("raw", "Could not parse model output."))
                    else:
                        st.write("**Strengths**")
                        for s in r.get("strengths", []):
                            st.write(f"- {s}")
                        st.write("**Gaps**")
                        for g in r.get("gaps", []):
                            st.write(f"- {g}")
