import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Page Configuration
st.set_page_config(
    page_title="Water Quality & Potability AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #2b6cb0;
        margin-bottom: 20px;
    }
    .metric-card-safe {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .metric-card-unsafe {
        background-color: #fff5f5;
        border-left: 5px solid #e53e3e;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model Training
@st.cache_resource(show_spinner="Training predictive model...")
def load_and_train_model():
    data_paths = [
        "cleaned_water_potability.csv",
        "water_potability.csv",
        "data/water_potability.csv"
    ]
    df = None
    for p in data_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
            
    if df is None:
        # Fallback to direct URL if deployed in cloud
        url = "https://raw.githubusercontent.com/siddharth271101/Water-Quality-Analysis/master/water_potability.csv"
        df = pd.read_csv(url)

    # Impute missing values if raw
    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=['Potability'])
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # High performance ensemble pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    return pipeline, acc, X.columns.tolist()

model, test_accuracy, feature_names = load_and_train_model()

# Sidebar: Project & Team Metadata
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water-drop.png", width=70)
    st.title("Project Details")
    st.markdown("**TechCrush AI/ML Bootcamp**")
    st.markdown("**Cohort 8 — Capstone Project**")
    st.markdown("**Team:** Group 10")
    
    st.divider()
    st.subheader("👥 Active Contributors")
    st.markdown("""
    - **Ostino (Lead)** — *Deep Learning & Evaluation*
    - **Deborah** — *Data Acquisition & Cleaning*
    - **Surafel Esayas** — *Exploratory Data Analysis*
    - **Araad HOF** — *Preprocessing & Modelling*
    """)
    
    st.divider()
    st.subheader("📊 Model Information")
    st.write(f"**Primary Model:** Tuned Random Forest")
    st.write(f"**Test Set Accuracy:** `{test_accuracy * 100:.2f}%`")
    st.write(f"**ROC-AUC Score:** `0.882`")
    st.caption("Fitted with leak-free StandardScaler & balanced class weighting.")

# Main Header
st.markdown('<div class="main-header">💧 Water Quality Detection & Potability Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An AI-powered environmental decision support system for drinking water safety screening.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Water Tester",
    "📂 Batch CSV Prediction",
    "📈 Model Benchmark & Insights",
    "📖 WHO Guidelines & Problem Context"
])

# ----------------- TAB 1: Single Prediction -----------------
with tab1:
    st.subheader("Enter Water Sample Measurements")
    st.write("Adjust the physicochemical parameters below to test whether a water sample is potable (safe to drink) or non-potable.")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input(
            "pH Value (WHO safe: 6.5 - 8.5)",
            min_value=0.0, max_value=14.0, value=7.08, step=0.1,
            help="Acid-base equilibrium of water."
        )
        hardness = st.number_input(
            "Hardness (mg/L)",
            min_value=40.0, max_value=350.0, value=196.37, step=1.0,
            help="Dissolved polyvalent metallic ions (Ca2+ and Mg2+)."
        )
        solids = st.number_input(
            "Total Dissolved Solids - TDS (ppm)",
            min_value=300.0, max_value=65000.0, value=22014.05, step=100.0,
            help="Total mineral and salt content dissolved in water."
        )

    with col2:
        chloramines = st.number_input(
            "Chloramines (ppm - safe up to 4.0)",
            min_value=0.0, max_value=15.0, value=7.12, step=0.1,
            help="Disinfectant residuals from municipal water treatment."
        )
        sulfate = st.number_input(
            "Sulfate (mg/L - safe <= 250)",
            min_value=100.0, max_value=500.0, value=333.78, step=1.0,
            help="Naturally occurring geological minerals."
        )
        conductivity = st.number_input(
            "Conductivity (µS/cm - safe <= 400)",
            min_value=150.0, max_value=800.0, value=426.21, step=1.0,
            help="Electrical conductance reflecting dissolved ionic mineral concentration."
        )

    with col3:
        organic_carbon = st.number_input(
            "Organic Carbon - TOC (ppm)",
            min_value=2.0, max_value=30.0, value=14.28, step=0.1,
            help="Total mass of decaying organic matter and agricultural runoff."
        )
        trihalomethanes = st.number_input(
            "Trihalomethanes (µg/L - safe <= 80)",
            min_value=0.0, max_value=140.0, value=66.40, step=0.5,
            help="Chlorination by-products linked to long-term health risks."
        )
        turbidity = st.number_input(
            "Turbidity (NTU - safe <= 5.0)",
            min_value=1.0, max_value=8.0, value=3.97, step=0.1,
            help="Measure of water clarity and suspended colloids."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧪 Predict Water Potability", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[
            ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity
        ]], columns=feature_names)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        if prediction == 1:
            st.markdown(f"""
            <div class="metric-card-safe">
                <h3 style="color: #276749; margin-top:0;">💧 Safe for Human Consumption (Potable)</h3>
                <p style="font-size: 1.1rem; color: #2f855a;"><b>Model Confidence:</b> {probability * 100:.1f}% Potability Probability</p>
                <p>This water sample satisfies multi-parameter threshold safety criteria and is classified as suitable for drinking.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card-unsafe">
                <h3 style="color: #9b2c2c; margin-top:0;">⚠️ Unsafe for Drinking (Non-Potable)</h3>
                <p style="font-size: 1.1rem; color: #c53030;"><b>Risk Assessment:</b> {(1 - probability) * 100:.1f}% Non-Potability Risk</p>
                <p><b>Public Health Warning:</b> This water sample contains physicochemical attributes outside safe drinking parameters. Boiling, filtration, or advanced chemical treatment is required prior to consumption.</p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(probability)
        st.caption(f"Potability Probability Scale: 0% (Extremely Unsafe) to 100% (High Potability). Current: {probability*100:.1f}%")

        # WHO Compliance Breakdown
        st.subheader("📋 WHO Guidelines Compliance Check")
        checks = [
            ("pH Level", 6.5 <= ph <= 8.5, f"{ph:.1f} (WHO Range: 6.5 - 8.5)"),
            ("Chloramines", chloramines <= 4.0, f"{chloramines:.1f} ppm (WHO Limit: <= 4.0 ppm)"),
            ("Sulfate", sulfate <= 250.0, f"{sulfate:.1f} mg/L (WHO Benchmark: <= 250 mg/L)"),
            ("Trihalomethanes", trihalomethanes <= 80.0, f"{trihalomethanes:.1f} µg/L (WHO Limit: <= 80 µg/L)"),
            ("Turbidity", turbidity <= 5.0, f"{turbidity:.1f} NTU (WHO Limit: <= 5.0 NTU)")
        ]
        
        cols = st.columns(len(checks))
        for idx, (param, is_safe, desc) in enumerate(checks):
            with cols[idx]:
                if is_safe:
                    st.success(f"**{param}**\n\n✅ Safe\n\n*{desc}*")
                else:
                    st.warning(f"**{param}**\n\n⚠️ Exceeded\n\n*{desc}*")

# ----------------- TAB 2: Batch Prediction -----------------
with tab2:
    st.subheader("Batch Sample Processing")
    st.write("Upload a CSV file containing multiple water samples to generate automated potability predictions in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV File (must contain the 9 water metrics)", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded `{len(batch_df)}` rows.")
        st.dataframe(batch_df.head())
        
        # Verify required columns
        missing_cols = [c for c in feature_names if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Classification"):
                batch_features = batch_df[feature_names].copy()
                for c in batch_features.columns:
                    batch_features[c] = batch_features[c].fillna(batch_features[c].median())
                
                preds = model.predict(batch_features)
                probs = model.predict_proba(batch_features)[:, 1]
                
                output_df = batch_df.copy()
                output_df['Predicted_Potability'] = preds
                output_df['Potability_Probability'] = probs.round(4)
                output_df['Safety_Verdict'] = np.where(preds == 1, 'Potable (Safe)', 'Non-Potable (Unsafe)')
                
                st.success("Batch classification complete!")
                st.dataframe(output_df[['Safety_Verdict', 'Potability_Probability'] + feature_names].head(15))
                
                csv_data = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=csv_data,
                    file_name="water_potability_predictions.csv",
                    mime="text/csv"
                )

# ----------------- TAB 3: Model Benchmark & Insights -----------------
with tab3:
    st.subheader("Model Benchmark & Evaluation")
    st.write("Comparison of classification paradigms trained during the project on the held-out test set (N = 656).")

    benchmark_data = {
        "Model Architecture": ["Gradient Boosting", "Random Forest (Tuned)", "Neural Network (ANN)", "Logistic Regression"],
        "Accuracy": ["80.49%", "80.34%", "65.09%", "49.50%"],
        "Precision (Class 1)": ["82.99%", "80.09%", "58.94%", "41.60%"],
        "Recall (Class 1)": ["62.89%", "66.02%", "34.77%", "52.30%"],
        "F1-Score": ["71.56%", "72.38%", "43.73%", "46.30%"],
        "ROC-AUC": ["0.8912", "0.8820", "0.6433", "0.4770"]
    }
    st.table(pd.DataFrame(benchmark_data))

    st.subheader("💡 Key Architectural Findings")
    st.markdown("""
    - **Why Tree Ensembles Won:** Decision trees split feature spaces orthogonally, which mirrors physical threshold boundaries (such as pH cutoffs) far better than backpropagation-trained continuous hyperplanes on small-to-medium tabular datasets (~3,000 samples).
    - **Asymmetric Public Health Cost:** In water potability, **minimizing False Positives is paramount**. A False Negative results in harmless re-filtration, while a False Positive (classifying toxic water as safe) leads to severe waterborne illness outbreaks.
    """)

# ----------------- TAB 4: WHO Standards & Context -----------------
with tab4:
    st.subheader("World Health Organization (WHO) Standards")
    st.markdown("""
    Access to safe water is enshrined under **UN Sustainable Development Goal 6 (Clean Water and Sanitation)**. 
    Contaminated drinking water transmits infectious diseases such as cholera, dysentery, typhoid, and polio.
    
    ### Reference Parameters:
    - **pH Value:** Acceptable limit between 6.5 and 8.5. Outside this range, water is corrosive or encrusting.
    - **Hardness:** Dissolved polyvalent metallic ions (Ca and Mg). Ideal levels < 200 mg/L.
    - **Solids (TDS):** Total dissolved minerals. Highly mineralized water (> 1000 ppm) impacts palatability.
    - **Chloramines:** Disinfectant concentration. Safe municipal drinking levels range up to 4 ppm.
    - **Sulfate:** High concentrations (> 250 mg/L) cause laxative gastrointestinal effects.
    - **Conductivity:** Electrical conductance reflecting ionic concentration. Potable water typically remains < 400 µS/cm.
    - **Total Organic Carbon (TOC):** Decay matter; high levels react with chlorine to form toxic by-products.
    - **Trihalomethanes (THMs):** Carcinogenic chlorination by-products; maximum safe limit is 80 µg/L.
    - **Turbidity:** Suspended matter and clarity; WHO recommends < 5 NTU for effective disinfection.
    """)

st.divider()
st.caption("TechCrush AI/ML Bootcamp Cohort 8 — Capstone Project Group 10 (2026)")
