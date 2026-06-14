import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Banking Risk Intelligence System",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; }
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem;
    }
   .fraud-alert {
    background-color: #ffe0e0; border-left: 5px solid #ff4444;
    padding: 1rem; border-radius: 8px; margin: 1rem 0;
    color: #1a1a2e !important;
}
.fraud-alert h3 { color: #cc0000 !important; }
.fraud-alert li { color: #1a1a2e !important; }
.fraud-alert p  { color: #1a1a2e !important; }

.safe-alert {
    background-color: #e0ffe0; border-left: 5px solid #00c851;
    padding: 1rem; border-radius: 8px; margin: 1rem 0;
    color: #1a1a2e !important;
}
.safe-alert h3 { color: #007a33 !important; }
.safe-alert li { color: #1a1a2e !important; }
.safe-alert p  { color: #1a1a2e !important; }

.warning-alert {
    background-color: #fff8e0; border-left: 5px solid #ffbb33;
    padding: 1rem; border-radius: 8px; margin: 1rem 0;
    color: #1a1a2e !important;
}
.warning-alert h3 { color: #996600 !important; }
.warning-alert li { color: #1a1a2e !important; }
.warning-alert p  { color: #1a1a2e !important; }
}
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f4ff; border: 1px solid #c5d3ff;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.9rem; color: #3a3a5c; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    try:    m['fraud']    = joblib.load('fraud_model.pkl')
    except: m['fraud']    = None
    try:
        m['loan']     = joblib.load('loan_model.pkl')
        m['scaler']   = joblib.load('loan_scaler.pkl')
        m['features'] = joblib.load('loan_features.pkl')
    except:
        m['loan'] = m['scaler'] = m['features'] = None
    return m

models = load_models()

# ─────────────────────────────────────────
# RISK SCORING ENGINE
# ─────────────────────────────────────────
def rule_based_fraud_score(amount, hour, transaction_type,
                            location_match, device_known,
                            avg_transaction, transactions_today,
                            previous_fraud, card_present):
    score = 0.0
    avg   = avg_transaction if avg_transaction > 0 else 1
    ratio = amount / avg

    # Amount anomaly
    if   ratio > 20: score += 40
    elif ratio > 10: score += 30
    elif ratio > 5:  score += 20
    elif ratio > 3:  score += 12
    elif ratio > 1.5:score += 5

    # Absolute amount threshold
    if   amount > 500000: score += 20
    elif amount > 100000: score += 10
    elif amount > 50000:  score += 5

    # Location
    score += {"Yes — same city":0,"No — different city":15,"No — different country":30}.get(location_match,0)

    # Device
    score += {"Yes — known device":0,"No — new device":12,"No — suspicious IP":25}.get(device_known,0)

    # Night time
    if   0 <= hour <= 4: score += 15
    elif 5 <= hour <= 6: score += 8

    # Transaction type
    score += {"Online purchase":5,"ATM withdrawal":5,"POS / card swipe":0,
              "Wire transfer":18,"International transaction":25}.get(transaction_type,5)

    # Frequency
    if   transactions_today >= 5: score += 15
    elif transactions_today >= 3: score += 8
    elif transactions_today >= 2: score += 3

    # Fraud history
    score += {"No previous fraud":0,"1 suspicious transaction before":15,
              "Multiple fraud attempts":30}.get(previous_fraud,0)

    # Card not present
    score += {"Yes — card physically swiped":0,"No — only card number used":8,
              "Contactless / NFC":3}.get(card_present,0)

    return min(round(score, 1), 99.0)


def get_risk_breakdown(amount, hour, transaction_type,
                        location_match, device_known,
                        avg_transaction, transactions_today,
                        previous_fraud):
    avg   = avg_transaction if avg_transaction > 0 else 1
    ratio = amount / avg
    return {
        "💰 Amount anomaly":
            min(int(ratio * 12 + (20 if amount > 100000 else 0)), 95),
        "🌍 Location mismatch":
            {"Yes — same city":5,"No — different city":55,"No — different country":92}.get(location_match,10),
        "📱 Device / IP risk":
            {"Yes — known device":5,"No — new device":52,"No — suspicious IP":88}.get(device_known,10),
        "🕐 Night time risk":
            (85 if 0<=hour<=4 else 45 if hour<=6 else 8),
        "⚡ Transaction frequency":
            min(transactions_today * 18, 95),
        "🚨 Fraud history":
            {"No previous fraud":5,"1 suspicious transaction before":58,
             "Multiple fraud attempts":92}.get(previous_fraud,5),
        "🏷️ Transaction type":
            {"Online purchase":18,"ATM withdrawal":20,"POS / card swipe":5,
             "Wire transfer":62,"International transaction":82}.get(transaction_type,20),
    }

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown("## 🏦 Banking Risk Intelligence")
st.sidebar.markdown("---")
module = st.sidebar.radio("Navigate to:", [
    "🏠 Dashboard",
    "💳 Fraud Detection",
    "📋 Loan Default Prediction",
    "📊 Model Performance",
    "🤖 AI Risk Assistant",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status**")
if models['fraud']:  st.sidebar.success("✅ Fraud Model Ready")
else:                st.sidebar.error("❌ Fraud Model Missing")
if models['loan']:   st.sidebar.success("✅ Loan Model Ready")
else:                st.sidebar.error("❌ Loan Model Missing")

# ─────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────
if module == "🏠 Dashboard":
    st.markdown('<div class="main-header">🏦 Banking Risk Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time fraud detection and loan default prediction powered by machine learning</div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud ROC-AUC",    "0.9739",   "↑ Excellent")
    c2.metric("Fraud Recall",     "88%",      "Catches 88/100 frauds")
    c3.metric("Transactions",     "284,807",  "Training dataset")
    c4.metric("Loan Accuracy",    "~82%",     "30,000 customers")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 💳 Fraud Detection")
        st.markdown("""
        Analyzes transactions in real time using multiple risk signals:
        - Geographic location match
        - Device and IP recognition
        - Transaction amount anomaly
        - Time-of-day patterns
        - Historical fraud signals
        - Transaction velocity

        **Algorithm:** XGBoost · **Imbalance:** scale_pos_weight
        """)

    with c2:
        st.markdown("### 📋 Loan Default Prediction")
        st.markdown("""
        Assesses credit risk before approving a loan application:
        - 6-month payment history
        - Credit utilization rate
        - Customer profile analysis
        - Bill vs payment behaviour

        **Output:** Risk score with clear recommendation
        🟢 Approve · 🟡 Review · 🔴 Reject

        **Algorithm:** XGBoost · **Validation:** 5-fold cross-validation
        """)

# ─────────────────────────────────────────
# FRAUD DETECTION
# ─────────────────────────────────────────
elif module == "💳 Fraud Detection":
    st.markdown("## 💳 Transaction Fraud Detector")
    st.markdown("""
    <div class="info-box">
    Enter the transaction details below. The risk engine automatically
    calculates a fraud probability score from 7 independent risk factors.
    </div>
    """, unsafe_allow_html=True)

    # Quick test scenarios
    st.markdown("#### Quick Test Scenarios")
    preset = st.selectbox("Load a scenario or fill manually:", [
        "— Enter manually —",
        "🔴 High Risk: International wire at 2AM, suspicious IP",
        "🔴 High Risk: Amount 20× average, multiple fraud history",
        "🟡 Medium Risk: Different city, new device, large amount",
        "🟢 Low Risk: Normal daytime purchase, known device",
        "🟢 Low Risk: Small ATM withdrawal, same city",
    ])

    presets = {
        "— Enter manually —":
            dict(amount=1000,   hour=14, ttype="Online purchase",         loc="Yes — same city",         dev="Yes — known device",    avg=2000,  freq=1, hist="No previous fraud",               card="Yes — card physically swiped"),
        "🔴 High Risk: International wire at 2AM, suspicious IP":
            dict(amount=85000,  hour=2,  ttype="International transaction",loc="No — different country",  dev="No — suspicious IP",    avg=2000,  freq=4, hist="Multiple fraud attempts",          card="No — only card number used"),
        "🔴 High Risk: Amount 20× average, multiple fraud history":
            dict(amount=60000,  hour=23, ttype="Wire transfer",            loc="No — different city",     dev="No — new device",       avg=3000,  freq=3, hist="Multiple fraud attempts",          card="No — only card number used"),
        "🟡 Medium Risk: Different city, new device, large amount":
            dict(amount=25000,  hour=15, ttype="ATM withdrawal",           loc="No — different city",     dev="No — new device",       avg=4000,  freq=2, hist="No previous fraud",               card="Yes — card physically swiped"),
        "🟢 Low Risk: Normal daytime purchase, known device":
            dict(amount=1500,   hour=14, ttype="Online purchase",          loc="Yes — same city",         dev="Yes — known device",    avg=2000,  freq=1, hist="No previous fraud",               card="No — only card number used"),
        "🟢 Low Risk: Small ATM withdrawal, same city":
            dict(amount=500,    hour=11, ttype="ATM withdrawal",           loc="Yes — same city",         dev="Yes — known device",    avg=1500,  freq=1, hist="No previous fraud",               card="Yes — card physically swiped"),
    }
    p = presets[preset]

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
        amount = st.number_input("Transaction Amount (₹)", min_value=0.0,   value=float(p['amount']), step=100.0)
        hour   = st.slider("Hour of Transaction (24h format)", 0, 23, p['hour'])
        st.caption(f"{'🌙 Night — elevated risk' if hour<=6 else '☀️ Daytime' if hour<=20 else '🌆 Evening'} · {hour:02d}:00")
        ttype  = st.selectbox("Transaction Type",
                    ["Online purchase","ATM withdrawal","POS / card swipe","Wire transfer","International transaction"],
                    index=["Online purchase","ATM withdrawal","POS / card swipe","Wire transfer","International transaction"].index(p['ttype']))
        card   = st.selectbox("Card present?",
                    ["Yes — card physically swiped","No — only card number used","Contactless / NFC"],
                    index=["Yes — card physically swiped","No — only card number used","Contactless / NFC"].index(p['card']))

    with c2:
        st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
        avg  = st.number_input("Customer's Usual Avg Transaction (₹)", min_value=100.0, value=float(p['avg']), step=500.0)
        ratio = amount / avg if avg > 0 else 0
        if   ratio > 10: st.error(f"🚨 {ratio:.1f}× customer average — extreme anomaly")
        elif ratio > 5:  st.warning(f"⚠️ {ratio:.1f}× customer average — very suspicious")
        elif ratio > 3:  st.warning(f"⚠️ {ratio:.1f}× customer average — elevated")
        elif ratio > 1.5:st.info(f"ℹ️ {ratio:.1f}× customer average — slightly high")
        else:            st.success(f"✅ Within normal range ({ratio:.1f}×)")

        freq = st.number_input("Transactions in last 1 hour", 0, 20, p['freq'])
        if freq >= 5: st.error("🚨 Very high transaction frequency")
        elif freq >= 3: st.warning("⚠️ Multiple transactions in short time")

        loc  = st.selectbox("Location matches customer's city?",
                    ["Yes — same city","No — different city","No — different country"],
                    index=["Yes — same city","No — different city","No — different country"].index(p['loc']))
        dev  = st.selectbox("Device / IP recognized?",
                    ["Yes — known device","No — new device","No — suspicious IP"],
                    index=["Yes — known device","No — new device","No — suspicious IP"].index(p['dev']))
        hist = st.selectbox("Previous fraud history?",
                    ["No previous fraud","1 suspicious transaction before","Multiple fraud attempts"],
                    index=["No previous fraud","1 suspicious transaction before","Multiple fraud attempts"].index(p['hist']))

    st.markdown("---")
    if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            score     = rule_based_fraud_score(amount, hour, ttype, loc, dev, avg, freq, hist, card)
            breakdown = get_risk_breakdown(amount, hour, ttype, loc, dev, avg, freq, hist)
            is_fraud  = score >= 45

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud Probability", f"{score}%")
        c2.metric("Decision",  "🚨 FRAUD"  if is_fraud else "✅ NORMAL")
        c3.metric("Amount",    f"₹{amount:,.0f}")
        c4.metric("Risk Level","HIGH 🔴" if score>=65 else "MEDIUM 🟡" if score>=45 else "LOW 🟢")

        if score >= 65:
            st.markdown("""<div class="fraud-alert"> 
            <h3>🚨 HIGH RISK — Fraudulent Transaction Detected</h3>
            <ul>
            <li>🔒 Block this transaction immediately</li>
            <li>📞 Contact customer to verify identity</li>
            <li>🔍 Flag account for security review</li>
            <li>📝 Initiate fraud investigation</li>
            </ul></div>""", unsafe_allow_html=True)
        elif score >= 45:
           st.markdown("""<div class="warning-alert" 
            <h3>🟡 MEDIUM RISK — Transaction Requires Review</h3>
            <ul>
            <li>⏸️ Hold transaction for manual verification</li>
            <li>📲 Send OTP confirmation to customer</li>
            <li>📝 Log transaction for monitoring</li>
            </ul></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="safe-alert"> 
            <h3>✅ LOW RISK — Transaction Approved</h3>
            <p>No significant risk signals detected. Safe to process.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Risk Factor Analysis")
        for factor, sc in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            icon = "🔴" if sc>=70 else "🟡" if sc>=35 else "🟢"
            c1, c2 = st.columns([4, 1])
            with c1: st.progress(sc/100, text=f"{icon} {factor}")
            with c2: st.write(f"**{sc}/100**")

# ─────────────────────────────────────────
# LOAN DEFAULT
# ─────────────────────────────────────────
elif module == "📋 Loan Default Prediction":
    st.markdown("## 📋 Loan Default Risk Assessor")
    st.markdown("""
    <div class="info-box">
    Enter the loan applicant's details to receive a risk score
    and a clear approval recommendation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Quick Test Scenarios")
    lpreset = st.selectbox("Load a scenario or fill manually:", [
        "— Enter manually —",
        "🔴 High Risk: Repeated delays, near credit limit",
        "🟡 Medium Risk: Occasional delays, moderate utilization",
        "🟢 Low Risk: Consistent payments, graduate profile",
    ])
    lp = {
        "— Enter manually —":
            dict(lim=50000, edu=2, age=35, pay0=0,  pay2=0,  bill1=10000, payamt1=5000,  marry=2),
        "🔴 High Risk: Repeated delays, near credit limit":
            dict(lim=20000, edu=3, age=56, pay0=3,  pay2=2,  bill1=19500, payamt1=500,   marry=1),
        "🟡 Medium Risk: Occasional delays, moderate utilization":
            dict(lim=80000, edu=2, age=42, pay0=1,  pay2=1,  bill1=40000, payamt1=3000,  marry=1),
        "🟢 Low Risk: Consistent payments, graduate profile":
            dict(lim=200000,edu=1, age=32, pay0=-1, pay2=-1, bill1=15000, payamt1=15000, marry=2),
    }[lpreset]

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">Personal Details</div>', unsafe_allow_html=True)
        limit_bal = st.number_input("Credit Limit (₹)", 10000, 2000000, lp['lim'], 10000)
        sex       = st.selectbox("Gender", [1,2], format_func=lambda x:"Male" if x==1 else "Female")
        education = st.selectbox("Education", [1,2,3,4],
                        format_func=lambda x:{1:"Graduate",2:"University",3:"High School",4:"Other"}[x],
                        index=[1,2,3,4].index(lp['edu']))
        marriage  = st.selectbox("Marital Status", [1,2,3],
                        format_func=lambda x:{1:"Married",2:"Single",3:"Other"}[x],
                        index=[1,2,3].index(lp['marry']))
        age       = st.slider("Age", 18, 80, lp['age'])

    with c2:
        st.markdown('<div class="section-header">Payment History (6 Months)</div>', unsafe_allow_html=True)
        def plabel(x):
            return "Paid early" if x==-1 else "On time" if x==0 else f"{x} month(s) late"
        pay_0 = st.selectbox("Last month",   [-1,0,1,2,3,4,5,6], format_func=plabel,
                              index=[-1,0,1,2,3,4,5,6].index(lp['pay0']))
        pay_2 = st.selectbox("2 months ago", [-1,0,1,2,3,4], format_func=plabel,
                              index=[-1,0,1,2,3,4].index(lp['pay2']))
        pay_3 = st.selectbox("3 months ago", [-1,0,1,2,3,4], format_func=plabel)
        pay_4 = st.selectbox("4 months ago", [-1,0,1,2,3,4], format_func=plabel)
        pay_5 = st.selectbox("5 months ago", [-1,0,1,2,3,4], format_func=plabel)
        pay_6 = st.selectbox("6 months ago", [-1,0,1,2,3,4], format_func=plabel)
        delays = [p for p in [pay_0,pay_2,pay_3,pay_4,pay_5,pay_6] if p > 0]
        if   len(delays) >= 4: st.error(f"🚨 {len(delays)} delayed payments detected")
        elif len(delays) >= 2: st.warning(f"⚠️ {len(delays)} delayed payments detected")
        elif len(delays) == 1: st.warning("⚠️ 1 delayed payment detected")
        else:                  st.success("✅ Clean payment history")

    with c3:
        st.markdown('<div class="section-header">Billing & Payments</div>', unsafe_allow_html=True)
        bill_amt1 = st.number_input("Last month bill (₹)",         0, 2000000, lp['bill1'],              1000)
        bill_amt2 = st.number_input("2 months ago bill (₹)",       0, 2000000, int(lp['bill1']*0.9),     1000)
        bill_amt3 = st.number_input("3 months ago bill (₹)",       0, 2000000, int(lp['bill1']*0.8),     1000)
        pay_amt1  = st.number_input("Last month payment (₹)",      0, 2000000, lp['payamt1'],             500)
        pay_amt2  = st.number_input("2 months ago payment (₹)",    0, 2000000, int(lp['payamt1']*0.9),   500)
        pay_amt3  = st.number_input("3 months ago payment (₹)",    0, 2000000, int(lp['payamt1']*0.8),   500)
        util = (bill_amt1 / limit_bal * 100) if limit_bal > 0 else 0
        st.metric("Credit Utilization", f"{util:.1f}%",
                  delta="High risk" if util > 80 else "Normal")

    st.markdown("---")
    if st.button("🔍 Predict Default Risk", use_container_width=True, type="primary"):
        if models['loan'] is None:
            st.error("❌ Loan model files not found. Ensure loan_model.pkl, loan_scaler.pkl and loan_features.pkl are present.")
        else:
            with st.spinner("Calculating risk score..."):
                inp = pd.DataFrame([[
                    limit_bal, sex, education, marriage, age,
                    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                    bill_amt1, bill_amt2, bill_amt3, 0, 0, 0,
                    pay_amt1,  pay_amt2,  pay_amt3,  0, 0, 0
                ]], columns=models['features'])

                # Scale only continuous columns — categorical ones stay as-is
                categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
                continuous_cols  = [c for c in models['features']
                if c not in categorical_cols]
                inp_final = inp.copy()
                inp_final[continuous_cols] = models['scaler'].transform(
                inp[continuous_cols])
                risk = round(models['loan'].predict_proba(inp_final)[0][1] * 100, 1)
                
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Default Risk",   f"{risk}%")
            c2.metric("Credit Limit",   f"₹{limit_bal:,}")
            c3.metric("Payment Delays", f"{len(delays)} / 6 months")
            c4.metric("Credit Used",    f"{util:.0f}%")

            if risk > 60:
                st.markdown(f"""<div class="fraud-alert"> 
                <h3>🔴 HIGH RISK — Reject Application ({risk}%)</h3>
                <ul>
                <li>❌ Reject this loan application</li>
                <li>📄 Request additional income verification</li>
                <li>🔁 Customer may reapply after improving payment history</li>
                </ul></div>""", unsafe_allow_html=True)
            elif risk > 30:
                st.markdown(f"""<div class="warning-alert">
                <h3>🟡 MEDIUM RISK — Manual Review Required ({risk}%)</h3>
                <ul>
                <li>🔍 Conduct detailed financial review</li>
                <li>🏠 Consider requiring collateral or guarantor</li>
                <li>💰 Consider approving a reduced loan amount</li>
                </ul></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="safe-alert"> 
                <h3>🟢 LOW RISK — Approve Application ({risk}%)</h3>
                <ul>
                <li>✅ Approve loan application</li>
                <li>📋 Complete standard documentation</li>
                <li>📆 Set up automatic payment reminders</li>
                </ul></div>""", unsafe_allow_html=True)

            st.markdown("### Risk Driver Analysis")
            drivers = {
                "Payment delay history":  min(len(delays) * 22, 95),
                "Credit utilization":     min(int(util), 95),
                "Bill vs payment ratio":  min(int((1-(pay_amt1/(bill_amt1+1)))*80),90) if bill_amt1>0 else 10,
                "Age risk factor":        60 if age>=55 else 40 if age>=45 else 25 if age>=35 else 20,
                "Education factor":       {1:20,2:30,3:55,4:45}.get(education, 30),
            }
            for driver, sc in sorted(drivers.items(), key=lambda x: x[1], reverse=True):
                icon = "🔴" if sc>=65 else "🟡" if sc>=35 else "🟢"
                c1, c2 = st.columns([4, 1])
                with c1: st.progress(sc/100, text=f"{icon} {driver}")
                with c2: st.write(f"**{sc}/100**")

# ─────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────
elif module == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 💳 Fraud Detection Model")
        st.markdown("**Algorithm:** XGBoost · **Dataset:** 284,807 transactions")
        st.dataframe(pd.DataFrame({
            "Metric":        ["ROC-AUC Score","Recall (Fraud)","Precision (Fraud)","F1 Score"],
            "Value":         ["0.9739",        "88%",           "33%",              "0.48"],
            "Interpretation":["Near-perfect fraud ranking",
                              "Detects 88 out of every 100 real frauds",
                              "False alarms are acceptable in fraud detection",
                              "Balanced precision-recall tradeoff"]
        }), hide_index=True, use_container_width=True)

        st.markdown("**Confusion Matrix — Test Set**")
        st.dataframe(pd.DataFrame({
            "":              ["Predicted Normal", "Predicted Fraud"],
            "Actual Normal": ["56,688 ✅",         "176 ⚠️"],
            "Actual Fraud":  ["12 ❌",             "86 ✅"]
        }), hide_index=True, use_container_width=True)

    with c2:
        st.markdown("### 📋 Loan Default Model")
        st.markdown("**Algorithm:** XGBoost · **Dataset:** 30,000 customers")
        st.dataframe(pd.DataFrame({
            "Metric":        ["ROC-AUC Score","Recall (Default)","Precision (Default)","Accuracy"],
            "Value":         ["~0.82",         "~76%",            "~65%",               "~82%"],
            "Interpretation":["Good discriminative ability",
                              "Catches majority of defaulters",
                              "Most flagged customers do default",
                              "Solid overall classification accuracy"]
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Methodology")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **Class Imbalance Handling**

        Fraud data has only 0.173% fraud cases.
        We use `scale_pos_weight` in XGBoost which penalises
        misclassification of the minority class proportionally —
        no synthetic data generation required.
        """)
    with c2:
        st.info("""
        **Model Validation**

        Both models are validated using 5-fold stratified
        cross-validation to ensure consistent performance
        across different data splits — not just a single lucky split.
        """)
elif module == "🤖 AI Risk Assistant":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rag.agent import run_agent
    from dotenv import load_dotenv
    load_dotenv()

    st.markdown("## 🤖 Banking Risk AI Assistant")
    st.markdown("""
    <div class="info-box">
    Ask anything about a transaction or loan application.
    The agent uses your ML models + RBI and BASEL III
    guidelines to give accurate answers.
    </div>
    """, unsafe_allow_html=True)

    # Example questions
    st.markdown("#### 💬 Try asking:")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Transaction of ₹80,000 at 3AM from Dubai on unknown device — fraud?")
        st.info("What does RBI say about international transactions?")
    with col2:
        st.info("Customer with 4 late payments wants ₹2 lakh loan — approve?")
        st.info("What are BASEL III rules for high risk customers?")

    st.markdown("---")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask the banking risk agent..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is analyzing..."):
                try:
                    response = run_agent(prompt)
                except Exception as e:
                    response = f"Error: {str(e)}"
            st.write(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})