"""Streamlit interface for the RAG PDF chatbot (supports multiple PDFs)."""

import io
import os
from collections import defaultdict

import streamlit as st
from sentence_transformers import SentenceTransformer

from pdf_loader import load_pdf, split_into_chunks
from rag_engine import EMBEDDING_MODEL_NAME, NO_ANSWER_MESSAGE, RAGEngine

# Document used when the user has not uploaded any file
DEFAULT_PDF_PATH = "notes.pdf"

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄")


def get_api_key():
    """Read the API key from Streamlit secrets or from an environment variable."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load the embedding model once and share it across sessions."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Indexing the documents...")
def build_engine(api_key, documents):
    """Build one RAG engine over all documents; cached per set of documents."""
    engine = RAGEngine(api_key, embedding_model=load_embedding_model())

    all_chunks = []
    for file_name, file_bytes in documents:
        pages = load_pdf(io.BytesIO(file_bytes))
        all_chunks.extend(split_into_chunks(pages, source=file_name))

    if not all_chunks:
        return None

    engine.build_index(all_chunks)
    return engine


def format_sources(sources):
    """Group (file, page) pairs by file and return a readable string."""
    pages_by_file = defaultdict(list)
    for file_name, page in sources:
        pages_by_file[file_name].append(str(page))

    return "; ".join(
        f"{file_name} (pages {', '.join(pages)})"
        for file_name, pages in pages_by_file.items()
    )


st.title("📄 RAG PDF Chatbot")
st.caption("Ask questions about one or more PDFs. Answers are grounded in the documents and cite sources.")

api_key = get_api_key()
if not api_key:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()

# Sidebar: optional upload of multiple PDFs, otherwise use the default document
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    documents = tuple((f.name, f.getvalue()) for f in uploaded_files)
elif os.path.exists(DEFAULT_PDF_PATH):
    with open(DEFAULT_PDF_PATH, "rb") as f:
        documents = ((DEFAULT_PDF_PATH, f.read()),)
else:
    st.info("Upload one or more PDFs from the sidebar to get started.")
    st.stop()

st.sidebar.write("Current documents:")
for file_name, _ in documents:
    st.sidebar.write(f"- {file_name}")

engine = build_engine(api_key, documents)
if engine is None:
    st.error("No extractable text found. Scanned PDFs are not supported.")
    st.stop()

# Reset the chat history whenever the set of documents changes
document_names = tuple(name for name, _ in documents)
if st.session_state.get("document_names") != document_names:
    st.session_state.document_names = document_names
    st.session_state.messages = []

# Display the previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle a new question
question = st.chat_input("Ask a question about the documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer_text, sources = engine.answer(question)

        reply = answer_text
        # Show sources only when the answer comes from the documents
        if NO_ANSWER_MESSAGE.lower().rstrip(".") not in answer_text.lower():
            reply += f"\n\n*Sources: {format_sources(sources)}*"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})