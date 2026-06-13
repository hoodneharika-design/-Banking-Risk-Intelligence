import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from rag.retriever import search_documents
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = "load_dotenv()"

# ── Load models ──
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fraud_model   = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
loan_model    = joblib.load(os.path.join(BASE_DIR, "loan_model.pkl"))
loan_scaler   = joblib.load(os.path.join(BASE_DIR, "loan_scaler.pkl"))
loan_features = joblib.load(os.path.join(BASE_DIR, "loan_features.pkl"))

# ── Tool 1: Search banking guidelines ──
@tool
def search_banking_guidelines(query: str) -> str:
    """Search RBI, BASEL III and fraud pattern guidelines
    for regulatory context and recommended actions."""
    return search_documents(query)

# ── Tool 2: Analyze fraud risk ──
@tool
def analyze_fraud_risk(
    amount: float,
    hour: int,
    location_match: str,
    device_known: str,
    transaction_type: str
) -> str:
    """Analyze a bank transaction for fraud risk.
    Returns fraud probability and risk level.
    location_match: 'same city', 'different city', or 'different country'
    device_known: 'known', 'new', or 'suspicious'
    transaction_type: 'online', 'atm', 'pos', 'wire', or 'international'
    """
    features = np.zeros(29)

    features[0] = {
        "same city": 0.2,
        "different city": -2.5,
        "different country": -5.8
    }.get(location_match.lower(), 0)

    features[1] = {
        "known": 0.3,
        "new": -2.1,
        "suspicious": -4.9
    }.get(device_known.lower(), 0)

    features[2] = {
        "online": -0.8,
        "atm": -0.5,
        "pos": 0.2,
        "wire": -3.2,
        "international": -4.5
    }.get(transaction_type.lower(), 0)

    features[5]  = -3.8 if 0 <= hour <= 4 else 0.3
    features[13] = features[0] * 1.4
    features[28] = (amount - 88.35) / 250.12

    prob  = fraud_model.predict_proba([features])[0][1] * 100
    level = "HIGH RISK" if prob >= 65 else "MEDIUM RISK" if prob >= 45 else "LOW RISK"

    return f"Fraud Probability: {prob:.1f}% | Risk Level: {level}"

# ── Tool 3: Predict loan default ──
@tool
def predict_loan_default(
    credit_limit: float,
    age: int,
    education: int,
    last_payment_delay: int,
    bill_amount: float,
    payment_amount: float
) -> str:
    """Predict whether a loan applicant will default.
    education: 1=Graduate, 2=University, 3=High School, 4=Other
    last_payment_delay: 0=on time, 1=1 month late, 2=2 months late etc.
    Returns default probability and recommendation.
    """
    row = [
        credit_limit, 1, education, 2, age,
        last_payment_delay, 0, 0, 0, 0, 0,
        bill_amount, 0, 0, 0, 0, 0,
        payment_amount, 0, 0, 0, 0, 0
    ]

    inp    = pd.DataFrame([row], columns=loan_features)
    scaled = loan_scaler.transform(inp)
    prob   = loan_model.predict_proba(scaled)[0][1] * 100

    if prob > 60:
        rec = "REJECT — High default risk"
    elif prob > 30:
        rec = "REVIEW — Request additional documents"
    else:
        rec = "APPROVE — Low default risk"

    return f"Default Probability: {prob:.1f}% | Recommendation: {rec}"


# ── Create agent ──
def create_banking_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    tools = [
        search_banking_guidelines,
        analyze_fraud_risk,
        predict_loan_default
    ]
    agent = create_react_agent(
        llm,
        tools,
        prompt=(
            "You are an expert Banking Risk Intelligence Assistant. "
            "Help bank officers analyze fraud and loan default risk. "
            "Always use the available tools to get accurate data. "
            "Always cite which guideline or model output you base your decision on. "
            "Be concise, professional and specific with numbers."
        )
    )
    return agent


def run_agent(question: str) -> str:
    agent    = create_banking_agent()
    response = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    return response["messages"][-1].content


# ── Test ──
if __name__ == "__main__":
    print("\n🤖 Test 1 — Fraud Query")
    print(run_agent(
        "Transaction of Rs 85000 at 2AM from different country "
        "on suspicious IP. Is this fraud? What does RBI say?"
    ))

    print("\n🤖 Test 2 — Loan Query")
    print(run_agent(
        "Customer age 55, credit limit 20000, 3 months late payment, "
        "bill 19000, paid only 500. Should I approve the loan?"
    ))