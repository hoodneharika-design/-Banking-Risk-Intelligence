from fastapi import FastAPI
from pydantic import BaseModel

from rag.agent import run_agent

app = FastAPI(
    title="Banking Risk Intelligence API"
)

class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Banking Risk Intelligence API Running"
    }


@app.post("/agent/query")
def query_agent(request: QueryRequest):

    result = run_agent(request.question)

    return {
        "response": result
    }
