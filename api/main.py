from fastapi import FastAPI
from pydantic import BaseModel
from rag.agent import create_banking_agent, run_agent
from rag.retriever import search_documents

app = FastAPI(title="Banking Risk Intelligence API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Banking Risk Intelligence API is running ✅"}

@app.post("/agent/query")
def query_agent(request: QueryRequest):
    result = run_agent(request.question)
    return {"answer": result}

@app.post("/search/guidelines")
def search_guidelines(request: QueryRequest):
    result = search_documents(request.question)
    return {"results": result}

@app.get("/health")
def health():
    return {"status": "healthy"}