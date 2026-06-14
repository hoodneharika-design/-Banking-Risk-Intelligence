import os

# Document contents stored directly in code
# This bypasses ChromaDB completely — no version conflicts
DOCUMENTS = {
    "rbi": """
RBI FRAUD PREVENTION GUIDELINES 2023

1. TRANSACTION MONITORING
Banks must monitor all transactions above Rs 50,000 in real time.
Transactions originating from foreign IPs must be verified via OTP.
Multiple transactions within 1 hour from same account require review.
Night transactions between 12AM-5AM carry elevated fraud risk.

2. SUSPICIOUS TRANSACTION REPORTING
Banks must report suspicious transactions to FIU-IND within 7 days.
Any transaction deviating more than 300% from customer average must be flagged.
International transactions require additional KYC verification.

3. CARD FRAUD PREVENTION
Unrecognized device access must trigger immediate SMS alert to customer.
Transactions from new devices above Rs 10,000 require 2FA.
Location mismatch between card holder and transaction origin is a red flag.

4. LOAN DEFAULT PREVENTION
Customers with 3 or more consecutive late payments must be flagged.
Credit utilization above 80% indicates high default risk.
Customers in 60+ age group with high credit utilization need enhanced monitoring.
""",
    "fraud": """
COMMON BANKING FRAUD PATTERNS

1. CARD NOT PRESENT FRAUD
Fraudster uses stolen card details for online transactions.
Key indicators: New device, different IP, high amount, odd hours.
Recommended action: Block transaction, send OTP to registered mobile.

2. ACCOUNT TAKEOVER FRAUD
Fraudster gains access to customer account.
Key indicators: New device login, password change, large transfer within 24hrs.
Recommended action: Freeze account, contact customer immediately.

3. VELOCITY FRAUD
Multiple transactions in short time window.
Key indicators: 3+ transactions within 1 hour, escalating amounts.
Recommended action: Temporary hold after 3rd transaction, manual review.

4. INTERNATIONAL FRAUD
Card used internationally without prior intimation.
Key indicators: Foreign IP, currency mismatch, odd hours.
Recommended action: Block international transactions, verify with customer.

5. LOAN FRAUD INDICATORS
Fake income documents, inflated assets.
Key indicators: Very high credit limit requests, short employment history.
Payment history: PAY_0 >= 2 months late is strongest default predictor.
Recommended action: Request original salary slips, bank statements.
""",
    "basel": """
BASEL III CREDIT RISK GUIDELINES

1. CREDIT RISK ASSESSMENT
Banks must maintain Capital Adequacy Ratio (CAR) of minimum 10.5%.
All loan applications must undergo credit scoring using approved models.
Default probability above 60% requires automatic rejection workflow.
Portfolio-level default rate must be monitored monthly.

2. LOAN DEFAULT CLASSIFICATION
Standard: 0-30 days overdue
Sub-standard: 31-90 days overdue
Doubtful: 91-180 days overdue
Loss: 180+ days overdue

3. RISK WEIGHTED ASSETS
Retail loans: 75% risk weight
Mortgage loans: 35% risk weight
Corporate loans: 100% risk weight
High-risk customers: 150% risk weight

4. MODEL RISK MANAGEMENT
All ML models used for credit decisions must be validated annually.
Model explainability is mandatory for regulatory compliance.
SHAP values or equivalent must be maintained for audit trail.
Back-testing required every 6 months.
"""
}


def search_documents(query: str) -> str:
    """Simple keyword search across banking documents."""
    query_lower = query.lower()
    results     = []

    keywords = {
        "rbi":   ["rbi", "international", "transaction", "fraud", "suspicious",
                  "card", "device", "kyc", "otp", "monitor"],
        "fraud": ["fraud", "card", "device", "ip", "velocity", "account",
                  "takeover", "pattern", "international", "block"],
        "basel": ["basel", "credit", "risk", "loan", "default", "capital",
                  "adequacy", "risk weight", "high risk", "model"]
    }

    for doc_key, doc_keywords in keywords.items():
        if any(kw in query_lower for kw in doc_keywords):
            results.append(DOCUMENTS[doc_key])

    if not results:
        # Return all documents if no keyword match
        results = list(DOCUMENTS.values())

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    print(search_documents("international transactions RBI"))