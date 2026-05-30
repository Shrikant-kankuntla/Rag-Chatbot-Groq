import os
import re
import time
import shutil
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
CHROMA_PATH = os.path.join(BASE_DIR, "db")

PROMPT_TEMPLATE = """
You are a helpful assistant.

Rules:
- Answer ONLY from the provided context
- If answer is missing, say exactly:
  "Answer not found in the document."
- Do NOT guess
- Keep answer clear and concise
- Use bullet points if needed

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #0f172a;
}
.header {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg,#111827,#1f2937);
    color: white;
    margin-bottom: 18px;
}
.user-bubble {
    background: #2563eb;
    color: white;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    margin-left: auto;
}
.bot-bubble {
    background: #1f2937;
    color: white;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
}
section[data-testid="stSidebar"] {
    background: #020617;
}
.stButton button {
    border-radius: 10px;
}
.stDownloadButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h2>RAG Assistant</h2>
<p>Groq + ChromaDB + Streamlit</p>
</div>
""", unsafe_allow_html=True)


def is_greeting(query):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    return query.lower().strip() in greetings


def is_realtime_query(query):
    keywords = ["time", "today", "weather", "date", "current"]
    return any(k in query.lower() for k in keywords)


def safe_highlight(text, answer):
    try:
        for word in set(answer.split()):
            if len(word) > 5:
                safe = re.escape(word)
                text = re.sub(f"({safe})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
        return text
    except:
        return text


def format_chat_download(chat_history):
    lines = []
    for role, msg in chat_history:
        who = "User" if role == "user" else "Assistant"
        lines.append(f"{who}: {msg}")
    return "\n\n".join(lines)


def load_documents():
    docs = []
    if not os.path.exists(DATA_PATH):
        st.error(f"Data folder not found:\n{DATA_PATH}")
        return docs
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                except Exception:
                    pass
            elif file.endswith(".txt"):
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    docs.extend(loader.load())
                except Exception:
                    pass
            elif file.endswith(".md"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": file}))
                except Exception:
                    pass
    return docs


@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    docs = load_documents()
    if not docs:
        st.error("No documents found inside data folder.")
        st.stop()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    return db


def safe_retrieve(db, query):
    try:
        results = db.similarity_search_with_relevance_scores(query, k=5)
        docs = []
        with st.expander("Similarity Scores"):
            for doc, score in results:
                st.write(f"Score: {score}")
                st.write(doc.metadata)
                docs.append(doc)
        return docs
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []


@st.cache_resource
def load_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY missing in .env")
        st.stop()
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)
    return llm


db = load_db()
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.markdown("### Chat History")

if st.session_state.chat_history:
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.sidebar.markdown(f"**You:** {msg[:60]}{'...' if len(msg) > 60 else ''}")
        else:
            st.sidebar.markdown(f"**Bot:** {msg[:60]}{'...' if len(msg) > 60 else ''}")
else:
    st.sidebar.caption("No chat history yet.")

st.sidebar.markdown("---")

if st.sidebar.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.chat_history = []
    st.rerun()

chat_text = format_chat_download(st.session_state.chat_history)
st.sidebar.download_button(
    label="Download Chat",
    data=chat_text.encode("utf-8"),
    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg}</div>', unsafe_allow_html=True)

query = st.chat_input("Ask something from your documents...")

if query:
    start = time.time()
    st.session_state.chat_history.append(("user", query))
    st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)

    if is_greeting(query):
        resp = "Hello. Ask questions from your documents."
        st.session_state.chat_history.append(("assistant", resp))
        st.markdown(f'<div class="bot-bubble">{resp}</div>', unsafe_allow_html=True)
        st.stop()

    if is_realtime_query(query):
        resp = "Real-time information is not available."
        st.session_state.chat_history.append(("assistant", resp))
        st.markdown(f'<div class="bot-bubble">{resp}</div>', unsafe_allow_html=True)
        st.stop()

    docs = safe_retrieve(db, query)

    with st.expander("Retrieved Sources"):
        for d in docs:
            st.write(d.metadata)

    if not docs:
        resp = "Answer not found in the document."
        st.session_state.chat_history.append(("assistant", resp))
        st.markdown(f'<div class="bot-bubble">{resp}</div>', unsafe_allow_html=True)
        st.stop()

    context = "\n\n".join([d.page_content for d in docs])
    context = context[:5000]
    history = "\n".join(st.session_state.history[-6:])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    final_prompt = prompt.format(context=context, question=query, history=history)

    placeholder = st.empty()
    full = ""

    try:
        for chunk in model.stream(final_prompt):
            if chunk.content:
                full += chunk.content
                placeholder.markdown(f'<div class="bot-bubble">{full}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

    answer = full.strip()

    if len(answer) < 3 or "i don't know" in answer.lower():
        answer = "Answer not found in the document."

    st.session_state.history += [f"User: {query}", f"Bot: {answer}"]
    st.session_state.chat_history.append(("assistant", answer))

    with st.expander("Source Chunks"):
        for i, d in enumerate(docs):
            st.markdown(f"### Chunk {i+1}")
            st.code(d.page_content[:1000])

    with st.expander("Highlighted Context"):
        st.markdown(safe_highlight(context, answer), unsafe_allow_html=True)

    st.caption(f"Response Time: {round(time.time()-start,2)} sec")