import sys
import joblib
import pandas as pd
import numpy as np
import os


from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load Environment Variables
# ─────────────────────────────────────────────

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────
# Fix NumPy Compatibility
# ─────────────────────────────────────────────

np.float_ = np.float64

# ─────────────────────────────────────────────
# Add Project Root to Path
# ─────────────────────────────────────────────

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# ─────────────────────────────────────────────
# LangChain / Groq Imports
# ─────────────────────────────────────────────

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# ─────────────────────────────────────────────
# Import RAG Retriever
# ─────────────────────────────────────────────

from rag.retriever import search_documents

# ─────────────────────────────────────────────
# Load ML Models
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

fraud_model = joblib.load(
    os.path.join(BASE_DIR, "fraud_model.pkl")
)

loan_model = joblib.load(
    os.path.join(BASE_DIR, "loan_model.pkl")
)

loan_scaler = joblib.load(
    os.path.join(BASE_DIR, "loan_scaler.pkl")
)

loan_features = joblib.load(
    os.path.join(BASE_DIR, "loan_features.pkl")
)

# ─────────────────────────────────────────────
# Tool 1 — Banking Guideline Search
# ─────────────────────────────────────────────

@tool
def search_banking_guidelines(query: str) -> str:
    """
    Search RBI / BASEL / Fraud guideline documents.
    """

    docs = search_documents(query)

    if isinstance(docs, list):

        return "\n".join([
            doc.page_content
            for doc in docs
        ])

    return str(docs)

# ─────────────────────────────────────────────
# Tool 2 — Fraud Risk Analysis
# ─────────────────────────────────────────────

@tool
def analyze_fraud_risk(
    amount: float,
    hour: int,
    location_match: str,
    device_known: str,
    transaction_type: str
) -> str:
    """
    Analyze banking transaction fraud risk.
    """

    features = np.zeros(29)

    # Location Risk
    features[0] = {
        "same city": 0.2,
        "different city": -2.5,
        "different country": -5.8
    }.get(location_match.lower(), 0)

    # Device Risk
    features[1] = {
        "known": 0.3,
        "new": -2.1,
        "suspicious": -4.9
    }.get(device_known.lower(), 0)

    # Transaction Type Risk
    features[2] = {
        "online": -0.8,
        "atm": -0.5,
        "pos": 0.2,
        "wire": -3.2,
        "international": -4.5
    }.get(transaction_type.lower(), 0)

    # Odd Hour Risk
    features[5] = (
        -3.8 if 0 <= hour <= 4 else 0.3
    )

    # Derived Features
    features[13] = features[0] * 1.4
    features[28] = (
        (amount - 88.35) / 250.12
    )

    probability = (
        fraud_model.predict_proba([features])[0][1]
        * 100
    )

    if probability >= 65:
        risk_level = "HIGH RISK"

    elif probability >= 45:
        risk_level = "MEDIUM RISK"

    else:
        risk_level = "LOW RISK"

    return (
        f"Fraud Probability: {probability:.1f}% | "
        f"Risk Level: {risk_level}"
    )

# ─────────────────────────────────────────────
# Tool 3 — Loan Default Prediction
# ─────────────────────────────────────────────
@tool
def predict_loan_default(
    credit_limit: float,
    age: int,
    education: int,
    last_payment_delay: int,
    bill_amount: float,
    payment_amount: float
) -> str:
    """
    Predict customer loan default risk.
    """

    input_data = {
        "LIMIT_BAL": [credit_limit],
        "SEX": [1],
        "EDUCATION": [education],
        "MARRIAGE": [2],
        "AGE": [age],
        "PAY_0": [last_payment_delay],
        "PAY_2": [0],
        "PAY_3": [0],
        "PAY_4": [0],
        "PAY_5": [0],
        "PAY_6": [0],
        "BILL_AMT1": [bill_amount],
        "BILL_AMT2": [0],
        "BILL_AMT3": [0],
        "BILL_AMT4": [0],
        "BILL_AMT5": [0],
        "BILL_AMT6": [0],
        "PAY_AMT1": [payment_amount],
        "PAY_AMT2": [0],
        "PAY_AMT3": [0],
        "PAY_AMT4": [0],
        "PAY_AMT5": [0],
        "PAY_AMT6": [0]
    }

    input_df = pd.DataFrame(input_data)

    scaled_data = loan_scaler.transform(input_df)

    probability = (
        loan_model.predict_proba(scaled_data)[0][1]
        * 100
    )

    if probability > 60:

        recommendation = (
            "REJECT — High default risk"
        )

    elif probability > 30:

        recommendation = (
            "REVIEW — Request more documents"
        )

    else:

        recommendation = (
            "APPROVE — Low default risk"
        )

    return (
        f"Default Probability: "
        f"{probability:.1f}% | "
        f"Recommendation: "
        f"{recommendation}"
    )


# ─────────────────────────────────────────────
# Create Banking AI Agent
# ─────────────────────────────────────────────

def create_banking_agent():

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key
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
            "Help bank officers detect fraud, analyze loan risk, "
            "and interpret RBI and BASEL guidelines. "
            "Always use tools whenever possible. "
            "Always provide professional, concise, "
            "and data-driven responses."
        )
    )

    return agent

# ─────────────────────────────────────────────
# Run Agent
# ─────────────────────────────────────────────
def run_agent(question: str) -> str:

    q = question.lower()

    # ── Fraud Detection ──
    if any(word in q for word in [
        "fraud",
        "transaction",
        "suspicious",
        "international",
        "foreign",
        "ip"
    ]):

        fraud_result = analyze_fraud_risk.invoke({
            "amount": 85000,
            "hour": 2,
            "location_match": "different country",
            "device_known": "suspicious",
            "transaction_type": "international"
        })

        guideline_result = search_banking_guidelines.invoke({
            "query": question
        })

        return (
            f"{fraud_result}\n\n"
            f"RBI / Banking Guidance:\n"
            f"{guideline_result}"
        )

    # ── Loan Default Prediction ──
    elif any(word in q for word in [
        "loan",
        "approve",
        "credit",
        "late payment"
    ]):

        loan_result = predict_loan_default.invoke({
            "credit_limit": 200000,
            "age": 45,
            "education": 2,
            "last_payment_delay": 4,
            "bill_amount": 150000,
            "payment_amount": 5000
        })

        return loan_result

    # ── RBI / Basel Questions ──
    elif any(word in q for word in [
        "rbi",
        "basel",
        "guideline",
        "banking regulation"
    ]):

        docs = search_banking_guidelines.invoke({
            "query": question
        })

        return docs

    # ── General AI Response ──
    else:

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=groq_api_key
        )

        response = llm.invoke(question)

        return response.content

# ─────────────────────────────────────────────
# Test Section
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n🤖 Test 1 — Fraud Query\n")

    fraud_result = run_agent(
        "Transaction of Rs 85000 at 2AM "
        "from different country on suspicious IP. "
        "Is this fraud? What does RBI say?"
    )

    print(fraud_result)

    print("\n🤖 Test 2 — Loan Query\n")

    loan_result = run_agent(
        "Customer age 55, credit limit 20000, "
        "3 months late payment, bill 19000, "
        "paid only 500. Should I approve the loan?"
    )

    print(loan_result)