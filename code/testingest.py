import os
import shutil

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "db")

FILES = {
    "india.md": "C:/Users/shrik/Desktop/Rag/data/books/india.md",
    "greek.txt": "C:/Users/shrik/Desktop/Rag/data/books/greek.txt",
    "rome.md": "C:/Users/shrik/Desktop/Rag/data/books/rome.md",
    "Clauses.pdf": "C:/Users/shrik/Desktop/Rag/data/books/Clauses.pdf",
}



def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())



def main():
    print(" Starting ingestion...")


    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Old database removed")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    all_chunks = []
    chunk_id = 0

    
    for source_name, file_path in FILES.items():

        print(f"\n Processing: {source_name}")

        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            continue

        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            print("   → Using PyPDFLoader")
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            print("   → Using TextLoader")

        documents = loader.load()

        if not documents:
            print(" No content extracted.")
            continue

        print(f"    Documents loaded: {len(documents)}")

       
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

      
        chunks = splitter.split_documents(documents)

        print(f"    Chunks created: {len(chunks)}")

   
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk_id,
                "source": source_name,
            }

            # Add page number if PDF
            if "page" in chunk.metadata:
                metadata["page"] = chunk.metadata["page"]

            chunk.metadata = metadata
            chunk_id += 1

        all_chunks.extend(chunks)

    if not all_chunks:
        print(" No chunks created. Exiting.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(" Saving to Chroma...")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(" Database successfully created!")

if __name__ == "__main__":
    main()
