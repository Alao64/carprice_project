import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import sklearn
st.sidebar.caption(f"sklearn version: {sklearn.__version__}")

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# ── Known model stats (from the last training run — update these two if you ever retrain) ──
MODEL_R2 = 0.9724
MODEL_RMSE = 1406.39
ENSEMBLE_MODELS = ["CatBoost", "XGBRegressor", "Random Forest"]

NUMERICAL_COLUMNS = ['DoorsNum', 'Owners', 'Warranty', 'Engine_Size', 'Weight',
                      'carlength', 'carwidth', 'monthly_mileage', 'peakrpm',
                      'Estimated_Mileage', 'Car_Age', 'TAge']
CATEGORICAL_COLUMNS = ['Model_Brand', 'Model_Type', 'Fuel_Type', 'Transmission', 'Condition', 'Color',
                        'Cruise', 'Leather_Seats', 'Heated_Seats', 'Navigation', 'Insurance',
                        'Service_History', 'Safety', 'Premium_Sound', 'Multimedia', 'Bluetooth',
                        'Wheel', 'Sunroof', 'Cylinder_Numbers', 'Credit_History']

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1E3A5F, #2E86C1);
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        color: white;
        margin-top: 10px;
    }
    .metric-box h1 { font-size: 2.6rem; margin: 0; }
    .metric-box .range { font-size: 1rem; margin: 6px 0 0; opacity: 0.9; }
    .metric-box .compare { font-size: 0.85rem; margin-top: 8px; opacity: 0.75; }
    .section-label { font-size: 0.78rem; font-weight: 600;
                     text-transform: uppercase; color: #888; letter-spacing: 1px; }

    /* Sidebar: compact */
    section[data-testid="stSidebar"] { padding-top: 0; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] h3 { font-size: 0.75rem; text-transform: uppercase;
                     letter-spacing: 0.5px; color: #888; margin-bottom: 4px; }
    section[data-testid="stSidebar"] p { font-size: 0.82rem; line-height: 1.4; margin-bottom: 8px; }
    .badge { display:inline-block; background:#1E3A5F; color:#cfe3f7; font-size:0.7rem;
             padding:3px 7px; border-radius:6px; margin:2px 3px 2px 0; }
    .stat-row { display:flex; justify-content:space-between; font-size:0.78rem; padding:3px 0; }
    .stat-row span:last-child { font-weight:700; color:#2E86C1; }

    /* Glowing predict button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #22c55e, #4ade80, #22c55e);
        background-size: 200% 100%;
        color: #04150a;
        font-weight: 700;
        border: none;
        animation: glow-pulse 2.2s ease-in-out infinite, sheen 3.2s linear infinite;
    }
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 8px 0px rgba(34,197,94,0.55); }
        50% { box-shadow: 0 0 24px 6px rgba(34,197,94,0.55); }
    }
    @keyframes sheen {
        0% { background-position: 0% 0; }
        100% { background-position: 200% 0; }
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "This car price predictor was trained on a used-car listings dataset covering "
        "brand, condition, mileage, and ownership history. The model was built by "
        "ensembling the top performing models: CatBoost, XGBoost, and Random Forest  "
        "then combined into a weighted voting ensemble based on their tuned RMSE. "
        "The top 5 features driving each individual prediction are also displayed below the estimate."
    )
    st.markdown(" ".join(f'<span class="badge">{m}</span>' for m in ENSEMBLE_MODELS), unsafe_allow_html=True)

    st.markdown("### Model performance")
    st.markdown(f'<div class="stat-row"><span>R² score</span><span>{MODEL_R2:.3f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-row"><span>RMSE</span><span>${MODEL_RMSE:,.0f}</span></div>', unsafe_allow_html=True)

    st.markdown("### Links")
    st.markdown("[GitHub repo](#)")   # placeholder — replace with your repo URL
    st.markdown("[LinkedIn](#)")      # placeholder — replace with your profile URL

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚗 Car Price Predictor")
st.markdown("Fill in the car details below to get an estimated price.")


@st.cache_data
def load_options():
    df = pd.read_csv("artifacts/data.csv")
    type_brand_map = (
        df.groupby("Model_Brand")["Model_Type"]
        .unique()
        .apply(sorted)
        .to_dict()
    )
    return type_brand_map


@st.cache_data
def load_full_data():
    return pd.read_csv("artifacts/data.csv")


type_brand_map = load_options()

# ── Form (tabbed) ────────────────────────────────────────────────────────────
with st.form("car_price_form"):
    tab_basic, tab_specs, tab_history, tab_features = st.tabs(
        ["🔑 Basic Info", "⚙️ Specs", "📋 History", "✨ Features"]
    )

    with tab_basic:
        c1, c2, c3 = st.columns(3)
        Model_Brand = c1.selectbox("Model Brand", options=sorted(type_brand_map.keys()))
        Model_Type = c2.selectbox("Model Type", options=type_brand_map[Model_Brand])
        Car_Age = c3.number_input("Car Age (years)", min_value=0, max_value=25, value=5)
        Fuel_Type = c1.selectbox("Fuel Type", ["Petrol", "Electric", "Hybrid"])
        Transmission = c2.selectbox("Transmission", ["Automatic", "Manual"])
        Condition = c3.selectbox("Condition", ["Good", "Fair", "Excellent"])
        Color = c1.selectbox("Color", ["Brown", "Black", "White", "Silver", "Beige", "Yellow"])

    with tab_specs:
        t1, t2, t3, t4 = st.columns(4)
        Engine_Size = t1.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0)
        DoorsNum = t2.number_input("Number of Doors", min_value=2, max_value=6, value=4)
        Weight = t3.number_input("Weight (kg)", min_value=500, max_value=5000, value=1500)
        carlength = t4.number_input("Car Length (mm)", min_value=2000, max_value=6000, value=4500)
        carwidth = t1.number_input("Car Width (mm)", min_value=1000, max_value=3000, value=1800)
        peakrpm = t2.number_input("Peak RPM", min_value=3000, max_value=10000, value=5500)
        Cylinder_Numbers = t3.selectbox("Cylinders", ['five', 'four', 'six', 'three', 'two'])
        Wheel = t4.selectbox("Wheels", ["Chrome", "Steel", "Alloy", "Forged", "Hubcaps"])

    with tab_history:
        u1, u2, u3 = st.columns(3)
        Owners = u1.number_input("Previous Owners", min_value=0, max_value=10, value=1)
        Estimated_Mileage = u2.number_input("Estimated Mileage (km)", min_value=0, value=50000)
        monthly_mileage = u3.number_input("Monthly Mileage (km)", min_value=0, value=1500)
        Warranty = u1.number_input("Warranty (years)", min_value=0, max_value=10, value=1)
        TAge = u2.selectbox("Tyre Age", [1, 2, 3, 4, 5])
        Credit_History = u3.selectbox("Credit History", ["Good", "Average", "Poor"])
        Insurance = u1.selectbox("Insurance", ["No insurance", "Comprehensive", "Third Party", "Collision"])
        Service_History = u2.selectbox("Service History", ["Full Service", "Partial Service", "No Service"])
        Safety = u3.selectbox("Safety Ratings", ["5 stars", "4 stars", "3 stars", "2 stars", "1 star", "Not rated/Unknown"])

    with tab_features:
        f1, f2, f3, f4 = st.columns(4)
        Cruise = f1.radio("Cruise Control", ["Yes", "No"], horizontal=True)
        Leather_Seats = f2.radio("Leather Seats", ["Yes", "No"], horizontal=True)
        Heated_Seats = f3.radio("Heated Seats", ["Yes", "No"], horizontal=True)
        Navigation = f4.radio("Navigation", ["Yes", "No"], horizontal=True)
        Premium_Sound = f1.radio("Premium Sound", ["Yes", "No"], horizontal=True)
        Multimedia = f2.radio("Multimedia", ["Yes", "No"], horizontal=True)
        Bluetooth = f3.radio("Bluetooth", ["Yes", "No"], horizontal=True)
        Sunroof = f4.radio("Sunroof", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("🔍 Predict Car Price", use_container_width=True)


# ── Helper: map SHAP feature names back to human-readable label + actual value ──
def get_top_shap_features(shap_values, feature_names, input_df, top_n=5):
    rows = []
    for val, name in zip(shap_values, feature_names):
        if name.startswith("num_pipeline__"):
            col = name.replace("num_pipeline__", "")
            if col in input_df.columns:
                display_val = input_df[col].values[0]
                rows.append((col, display_val, val))
        elif name.startswith("cat_pipeline__"):
            rest = name.replace("cat_pipeline__", "")
            for col in CATEGORICAL_COLUMNS:
                prefix = col + "_"
                if rest.startswith(prefix):
                    category = rest[len(prefix):]
                    # only keep this row if it's the category actually chosen
                    if str(input_df[col].values[0]) == category:
                        rows.append((col, category, val))
                    break

    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows[:top_n]


# ── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    try:
        data = CustomData(
            DoorsNum=DoorsNum, Owners=Owners, Warranty=Warranty,
            Engine_Size=Engine_Size, Weight=Weight, carlength=carlength,
            carwidth=carwidth, monthly_mileage=monthly_mileage, peakrpm=peakrpm,
            Estimated_Mileage=Estimated_Mileage, Car_Age=Car_Age,
            Model_Type=Model_Type, Model_Brand=Model_Brand, Fuel_Type=Fuel_Type,
            Transmission=Transmission, Condition=Condition, Color=Color, Cruise=Cruise,
            Leather_Seats=Leather_Seats, Heated_Seats=Heated_Seats,
            Navigation=Navigation, Insurance=Insurance,
            Service_History=Service_History, Safety=Safety,
            Premium_Sound=Premium_Sound, Multimedia=Multimedia,
            Bluetooth=Bluetooth, Wheel=Wheel, Sunroof=Sunroof,
            TAge=TAge, Cylinder_Numbers=Cylinder_Numbers,
            Credit_History=Credit_History,
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        result = pipeline.predict(df)
        price = result[0]
        low, high = price - MODEL_RMSE, price + MODEL_RMSE

        # Average-price comparison against the training data, same brand/type
        full_df = load_full_data()
        similar = full_df[full_df["Model_Brand"] == Model_Brand]
        if len(similar) >= 5:  # fall back to brand-only if too few exact-type matches
            same_type = similar[similar["Model_Type"] == Model_Type]
            if len(same_type) >= 5:
                similar = same_type
        avg_price = similar["Price"].mean() if len(similar) > 0 else None

        compare_line = ""
        if avg_price and avg_price > 0:
            pct_diff = (price - avg_price) / avg_price * 100
            direction = "above" if pct_diff >= 0 else "below"
            compare_line = (
                f'<p class="compare">{abs(pct_diff):.1f}% {direction} the average '
                f'for similar {Model_Brand.title()} {Model_Type.title()} listings</p>'
            )

        st.markdown(f"""
        <div class="metric-box">
            <h1>${price:,.2f}</h1>
            <p class="range">Estimated range: ${low:,.0f} – ${high:,.0f}</p>
            {compare_line}
        </div>
        """, unsafe_allow_html=True)

        # SHAP explanation
        shap_values, feature_names, base_value = pipeline.explain(df)
        top_features = get_top_shap_features(shap_values, feature_names, df, top_n=5)

        st.markdown(
            f"<p style='color:#888; font-size:0.85rem;'>"
            f"Baseline (average predicted price): <b>${base_value:,.2f}</b> — "
            f"the features below shift this specific prediction away from that baseline."
            f"</p>",
            unsafe_allow_html=True,
        )
        st.markdown("#### 🔍 Top 5 features driving this prediction")

        max_abs = max(abs(sv) for _, _, sv in top_features) if top_features else 1
        bar_rows = ""
        for col, val, shap_val in top_features:
            pct = max(8, int(abs(shap_val) / max_abs * 100))
            is_pos = shap_val >= 0
            bar_color = "linear-gradient(90deg,#22c55e,#4ade80)" if is_pos else "linear-gradient(90deg,#ef4444,#f87171)"
            glow_color = "rgba(34,197,94,0.55)" if is_pos else "rgba(239,68,68,0.55)"
            sign = "+" if is_pos else "−"
            label = f"{col.replace('_', ' ')}: {val}"

            bar_rows += (
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px; font-size:0.82rem;">'
                f'<span style="width:170px; flex-shrink:0; text-align:right; color:#888;">{label}</span>'
                f'<div style="flex:1; background:#22303d; border-radius:6px; height:16px; overflow:hidden;">'
                f'<div style="width:{pct}%; height:100%; border-radius:6px; background:{bar_color}; '
                f'box-shadow:0 0 10px 1px {glow_color}; animation: bar-glow 2.2s ease-in-out infinite;"></div>'
                f'</div>'
                f'<span style="width:70px; flex-shrink:0; color:#ccc;">{sign}${abs(shap_val):,.0f}</span>'
                f'</div>'
            )

        st.markdown(
            '<style>@keyframes bar-glow {0%, 100% { filter: brightness(1); } 50% { filter: brightness(1.35); }}</style>'
            + bar_rows,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")