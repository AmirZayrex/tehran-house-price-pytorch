"""
Streamlit UI for Housing Price Prediction
PyTorch Neural Network Model
"""

import streamlit as st
import numpy as np
import torch
import joblib

st.set_page_config(
    page_title="Tehran House Price Predictor",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.35) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 80%, rgba(168,85,247,0.30) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(14,165,233,0.20) 0%, transparent 70%),
        linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0f0c29 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 680px !important; }

.hero { text-align: center; padding: 40px 0 24px; }
.hero-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 0 24px rgba(168,85,247,0.6));
}
.hero h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: white !important;
    letter-spacing: -0.5px;
    margin: 0 0 8px !important;
}
.hero p {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(168,85,247,0.85);
    margin-bottom: 16px;
    display: block;
}

label, .stSelectbox label, .stNumberInput label, .stCheckbox label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 0.9rem !important;
}

.stCheckbox > label > div {
    border-color: rgba(255,255,255,0.3) !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #06b6d4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.4) !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.6) !important;
}

.result-card {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.2) 0%,
        rgba(168,85,247,0.2) 50%,
        rgba(6,182,212,0.15) 100%);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(168,85,247,0.35);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(168,85,247,0.15), inset 0 1px 0 rgba(255,255,255,0.2);
    margin-top: 8px;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(168,85,247,0.9);
    display: block;
    margin-bottom: 12px;
}
.result-price {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 6px;
}
.result-unit { font-size: 0.85rem; color: rgba(255,255,255,0.45); margin-top: 4px; }

.stat-row {
    display: flex;
    gap: 12px;
    justify-content: center;
    margin-top: 20px;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 50px;
    padding: 8px 18px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
}
.stat-pill span { color: white; font-weight: 600; }

.stAlert {
    border-radius: 12px !important;
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.35) !important;
    color: white !important;
}

hr { border-color: rgba(255,255,255,0.08) !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    from src.models.pytorch_model import get_model

    scaler = joblib.load("Artifacts/scaler.pkl")
    location_dict = joblib.load("Artifacts/location_dict.pkl")

    model = get_model(input_dim=9, hidden_dims=[64, 32], dropout_rate=0.1)
    model.load_state_dict(torch.load("Artifacts/pytorch_model.pth", map_location="cpu"))
    model.eval()

    return model, scaler, location_dict


st.markdown("""
<div class="hero">
    <span class="hero-icon">🏙️</span>
    <h1>Tehran House Price</h1>
    <p>Neural network estimate based on location, size, and features</p>
</div>
""", unsafe_allow_html=True)


try:
    model, scaler, location_dict = load_artifacts()
    addresses = sorted(location_dict.keys())
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

# ── Location ──
st.markdown('<span class="section-label">📍 Location</span>', unsafe_allow_html=True)
address = st.selectbox("Neighborhood", addresses, label_visibility="collapsed")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Property Details ──
st.markdown('<span class="section-label">📐 Property Details</span>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (m²)", min_value=20, max_value=1000, value=100, step=5)
    room = st.number_input("Rooms", min_value=1, max_value=10, value=2)
with col2:
    floor = st.number_input("Floor", min_value=0, max_value=50, value=2)
    year = st.number_input("Year of Construction", min_value=1300, max_value=1404, value=1395)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Amenities ──
st.markdown('<span class="section-label">✅ Amenities</span>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    elevator = st.checkbox("Elevator")
with c2:
    parking = st.checkbox("Parking")
with c3:
    warehouse = st.checkbox("Warehouse")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ──
if st.button("Estimate Price →"):
    current_year = 1404

    if year > current_year:
        st.error(f"Year of construction cannot be greater than {current_year}.")
        st.stop()

    age = current_year - year
    area_per_room = area / room
    loc_score = location_dict.get(address, np.mean(list(location_dict.values())))

    # Feature order must match training pipeline:
    # Elevator, Floor, Area, Parking, Room, Warehouse, AreaPerRoom, Age, LocationScore
    X = np.array([[
        int(elevator),
        floor,
        area,
        int(parking),
        room,
        int(warehouse),
        area_per_room,
        age,
        loc_score
    ]], dtype=np.float32)

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        log_price = model(X_tensor).item()

    price = np.expm1(log_price)
    price_b = price / 1_000_000_000

    st.markdown(f"""
    <div class="result-card">
        <span class="result-label">Estimated Price</span>
        <div class="result-price">{price_b:,.2f} B</div>
        <div class="result-unit">Toman &nbsp;·&nbsp; {price:,.0f} Toman</div>
        <div class="stat-row">
            <div class="stat-pill">📍 <span>{address}</span></div>
            <div class="stat-pill">📐 <span>{area} m²</span> · <span>{room} rooms</span></div>
            <div class="stat-pill">🏗️ Age <span>{age} yrs</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)