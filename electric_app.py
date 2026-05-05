import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Electric Bill Predictor",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #f0f4ff;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 700px !important;
    padding: 2rem 2rem 4rem 2rem !important;
}

.app-header { text-align: center; padding: 2rem 0 1.5rem 0; }
.app-icon { font-size: 52px; display: block; margin-bottom: 0.5rem; }
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem; font-weight: 700;
    color: #1e3a5f; margin: 0; line-height: 1.2;
}
.app-subtitle {
    font-size: 0.82rem; color: #64748b;
    margin-top: 0.4rem; letter-spacing: 1.5px; text-transform: uppercase;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    margin: 1.5rem 0; border-radius: 2px;
}
.section-label {
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #3b82f6; margin-bottom: 0.8rem; margin-top: 1.2rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #3b82f633, transparent);
}
.stNumberInput > div > div {
    background: #f8faff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
.stNumberInput > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
.stNumberInput input {
    color: #1e3a5f !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    background: transparent !important;
}
.stNumberInput button {
    color: #3b82f6 !important; background: #eff6ff !important; border: none !important;
}
.stNumberInput button:hover { background: #dbeafe !important; }
.stNumberInput label, .stSelectbox label {
    color: #475569 !important; font-size: 0.85rem !important; font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: #f8faff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important; color: #1e3a5f !important;
}
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; padding: 0.9rem 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    margin-top: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,0.4) !important;
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
}

/* Stats Row */
.stats-row { display: flex; gap: 12px; margin-top: 1rem; }
.stat-card {
    flex: 1; background: #ffffff;
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.8rem; text-align: center;
    box-shadow: 0 2px 6px rgba(59,130,246,0.06);
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1rem; font-weight: 700; color: #1e3a5f;
}
.stat-label {
    font-size: 0.68rem; color: #94a3b8;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;
}

/* Result Box */
.result-container {
    background: linear-gradient(135deg, #1e3a5f, #1e40af);
    border-radius: 16px; padding: 2rem; margin-top: 1.5rem;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,58,95,0.25);
}
.result-container::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #60a5fa, #93c5fd, #60a5fa);
}
.result-label {
    font-size: 0.72rem; letter-spacing: 3px; text-transform: uppercase;
    color: #93c5fd; font-family: 'Space Mono', monospace; margin-bottom: 0.75rem;
}
.result-amount {
    font-family: 'Space Mono', monospace;
    font-size: 3rem; font-weight: 700; color: #ffffff; line-height: 1; margin-bottom: 0.5rem;
}
.result-note { font-size: 0.78rem; color: #93c5fd; font-style: italic; margin-top: 0.5rem; }

/* Confidence Bar */
.confidence-wrapper { margin-top: 1.5rem; }
.confidence-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.4rem;
}
.confidence-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: #475569;
}
.confidence-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; font-weight: 700;
}
.confidence-bar-bg {
    background: #e2e8f0; border-radius: 99px; height: 10px; overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s ease;
}
.confidence-tag {
    display: inline-block; margin-top: 0.5rem;
    padding: 0.25rem 0.75rem; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1px;
}

/* Range Box */
.range-box {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-top: 1rem; display: flex; gap: 1rem;
    box-shadow: 0 2px 6px rgba(59,130,246,0.05);
}
.range-item { flex: 1; text-align: center; }
.range-value {
    font-family: 'Space Mono', monospace;
    font-size: 1rem; font-weight: 700; color: #1e3a5f;
}
.range-label { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

.app-footer {
    text-align: center; margin-top: 2.5rem;
    font-size: 0.7rem; color: #94a3b8;
    letter-spacing: 1px; font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def load_model():
    df = pd.read_csv("electric_bill.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = df.drop('Monthly_Bill_PHP', axis=1)
    y = df['Monthly_Bill_PHP']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

model, feature_cols = load_model()

# ── Header ──
st.markdown("""
<div class="app-header">
    <span class="app-icon">⚡</span>
    <div class="app-title">Electric Bill Predictor</div>
    <div class="app-subtitle">Random Forest Regression · AI 5.0</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Stats Cards ──
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">2,000</div>
        <div class="stat-label">Instances</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">7</div>
        <div class="stat-label">Features</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">96.3%</div>
        <div class="stat-label">Accuracy (R²)</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">300</div>
        <div class="stat-label">Trees</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ──
st.markdown('<div class="section-label">🔌 Appliance Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    num_appliances = st.number_input("Number of Appliances", min_value=1, max_value=20, value=8, step=1)
with col2:
    num_ref = st.number_input("Number of Refrigerators", min_value=1, max_value=5, value=1, step=1)
washing_machine = st.selectbox("Has Washing Machine?", options=["Yes", "No"])

st.markdown('<div class="section-label">⏱️ Daily Usage Hours</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    ac_hours = st.number_input("AC Hours Per Day", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
with col4:
    tv_hours = st.number_input("TV Hours Per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
lights_hours = st.number_input("Lights Hours Per Day", min_value=0.0, max_value=24.0, value=8.0, step=0.5)

st.markdown('<div class="section-label">👥 Household</div>', unsafe_allow_html=True)
num_people = st.number_input("Number of People in the House", min_value=1, max_value=15, value=4, step=1)

wm_enc = 1 if washing_machine == "Yes" else 0

# ── Predict ──
if st.button("⚡ PREDICT MONTHLY BILL"):
    input_data = pd.DataFrame([{
        "Num_Appliances":       num_appliances,
        "AC_Hours_Per_Day":     ac_hours,
        "Num_People":           num_people,
        "TV_Hours_Per_Day":     tv_hours,
        "Lights_Hours_Per_Day": lights_hours,
        "Num_Refrigerators":    num_ref,
        "Has_Washing_Machine":  wm_enc,
    }])

    predicted_bill = model.predict(input_data)[0]

    # ── Confidence from tree predictions ──
    tree_preds = np.array([tree.predict(input_data.values)[0] for tree in model.estimators_])
    std_dev    = tree_preds.std()
    cv         = (std_dev / predicted_bill) * 100  # coefficient of variation
    confidence = max(0, min(100, 100 - cv))        # higher cv = lower confidence

    # ── Range ──
    low  = predicted_bill - std_dev
    high = predicted_bill + std_dev

    # ── Color based on confidence ──
    if confidence >= 85:
        bar_color = "#22c55e"
        tag_color = "#dcfce7"
        tag_text_color = "#166534"
        tag_label = "🟢 HIGH CONFIDENCE"
    elif confidence >= 70:
        bar_color = "#f59e0b"
        tag_color = "#fef3c7"
        tag_text_color = "#92400e"
        tag_label = "🟡 MODERATE CONFIDENCE"
    else:
        bar_color = "#ef4444"
        tag_color = "#fee2e2"
        tag_text_color = "#991b1b"
        tag_label = "🔴 LOW CONFIDENCE"

    st.markdown(f"""
    <div class="result-container">
        <div class="result-label">✦ Estimated Monthly Electric Bill ✦</div>
        <div class="result-amount">&#8369;{predicted_bill:,.2f}</div>
        <div class="result-note">Based on your household usage inputs · Random Forest Model</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence Bar ──
    st.markdown(f"""
    <div class="confidence-wrapper">
        <div class="confidence-header">
            <span class="confidence-label">🎯 Prediction Confidence</span>
            <span class="confidence-pct" style="color:{bar_color}">{confidence:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{confidence:.1f}%; background:{bar_color};"></div>
        </div>
        <span class="confidence-tag" style="background:{tag_color}; color:{tag_text_color};">
            {tag_label}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Predicted Range ──
    st.markdown(f"""
    <div class="range-box">
        <div class="range-item">
            <div class="range-label">📉 Low Estimate</div>
            <div class="range-value">&#8369;{max(0,low):,.2f}</div>
        </div>
        <div class="range-item" style="border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0;">
            <div class="range-label">🎯 Predicted</div>
            <div class="range-value" style="color:#2563eb;">&#8369;{predicted_bill:,.2f}</div>
        </div>
        <div class="range-item">
            <div class="range-label">📈 High Estimate</div>
            <div class="range-value">&#8369;{high:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="app-footer">
    ⚡ ELECTRIC BILL PREDICTOR &nbsp;·&nbsp; RANDOM FOREST REGRESSION &nbsp;·&nbsp; AI 5.0
</div>
""", unsafe_allow_html=True)
