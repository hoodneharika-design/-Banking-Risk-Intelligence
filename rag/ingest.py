from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

def ingest_documents():
    print("📚 Loading banking documents...")

    # Load all .txt files from documents folder
    loader = DirectoryLoader(
        "rag/documents/",
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")

    # Create embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    print("✅ Vector database ready!")
    return vectorstore


if __name__ == "__main__":
    ingest_documents()