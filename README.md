# RAG Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to query PDF, TXT, and Markdown documents using semantic search and AI-powered responses. Built with Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, and Groq.

---

## Features

* Query PDF, TXT, and Markdown documents
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Fast responses powered by Groq LLM
* Conversational chat interface
* Source chunk inspection
* Similarity score visualization
* Download chat history
* Clean Streamlit UI

---

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Groq
* Sentence Transformers

---

## Project Structure

```text
project/
│
├── app/
│   ├── app.py
│   └── db/
│
├── data/
│   ├── Clauses.pdf
│   ├── greek.txt
│   ├── india.md
│   └── rome.md
│
├── .env
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Shrikant-kankuntla/Rag-Chatbot-Groq.git
cd rag-assistant
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Add Documents

Place supported files inside the `data` folder.

Supported formats:

* `.pdf`
* `.txt`
* `.md`

Example:

```text
data/
├── Clauses.pdf
├── greek.txt
├── india.md
└── rome.md
```

---

## Run the Application

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## How It Works

1. Documents are loaded from the `data` directory.
2. Text is split into chunks using LangChain.
3. Embeddings are generated using Hugging Face models.
4. Chunks are stored in ChromaDB.
5. Relevant chunks are retrieved based on the user's query.
6. Retrieved context is sent to the Groq LLM.
7. The model generates answers strictly from the retrieved document context.

---

## Example Questions

* What is termination without cause?
* Summarize the trade practices described in the documents.
* Compare ancient India and Rome.
* Explain the governance system mentioned in the text.
* List the important clauses in the agreement.

---

## Key Dependencies

```text
streamlit
langchain
langchain-community
langchain-chroma
langchain-huggingface
langchain-groq
chromadb
sentence-transformers
python-dotenv
pypdf
```

---

## Future Enhancements

* Document upload from the UI
* Persistent chat memory
* Source citations
* Multi-user support
* Hybrid search
* Authentication system

---

