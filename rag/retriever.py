import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # return top 3 relevant chunks
    )
    return retriever


def search_documents(query: str) -> str:
    try:
        retriever = load_retriever()
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No specific guidelines found for this query."
        results = "\n\n".join([doc.page_content for doc in docs])
        return results
    except Exception as e:
        return f"Guidelines search error: {str(e)}"


# Test it
if __name__ == "__main__":
    result = search_documents("What to do when transaction is from foreign IP?")
    print(result)