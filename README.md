# 💧 Water Quality Detection and Potability Prediction

**TechCrush AI/ML Bootcamp — Cohort 8 — Capstone Project (Group 10)**

An end-to-end Machine Learning and Deep Learning system for automated drinking water potability detection based on 9 physicochemical water quality metrics, coupled with an interactive real-time Streamlit web deployment.

---

## 👥 Project Team & Contributions (Group 10)
- **Ostino (Lead):** Deep Learning (Artificial Neural Network), Model Evaluation Benchmark, Streamlit Web App Architecture & Final Compilation.
- **Deborah:** Data Acquisition, Dataset Documentation, Missingness Audit & Class-Conditional Median Imputation.
- **Surafel Esayas:** Exploratory Data Analysis (EDA), Univariate Feature Distributions, Class Imbalance Visualization & Correlation Heatmap.
- **Araad HOF:** Data Preprocessing, Stratified 80/20 Train-Test Split, Leak-Free StandardScaler Pipeline & Tuned Random Forest (GridSearchCV).

---

## 🚀 Live Web Application
Test the live predictive model here: **[Streamlit Live App](https://share.streamlit.io)**

---

## 📊 Dataset & Features
The model analyzes 9 continuous physicochemical parameters benchmarked against World Health Organization (WHO) drinking water standards:
1. **pH Value:** Acid-base balance (WHO permissible range: 6.5 – 8.5)
2. **Hardness:** Concentration of calcium and magnesium salts (mg/L)
3. **Solids (TDS):** Total Dissolved Solids (ppm)
4. **Chloramines:** Disinfectant concentration (safe limit: up to 4 ppm)
5. **Sulfate:** Naturally occurring minerals (safe benchmark: <= 250 mg/L)
6. **Conductivity:** Electrical conductance (typical: < 400 µS/cm)
7. **Organic Carbon (TOC):** Decaying organic matter (ppm)
8. **Trihalomethanes:** Chlorination by-products (safe limit: <= 80 µg/L)
9. **Turbidity:** Measure of water clarity and suspended colloids (WHO limit: <= 5 NTU)

---

## 📈 Model Performance Benchmark (Test Set: N = 656)

| Model Architecture | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **80.49%** | **82.99%** | 62.89% | 71.56% | **0.8912** |
| **Random Forest (Tuned)** | 80.34% | 80.09% | **66.02%** | **72.38%** | 0.8820 |
| **Neural Network (ANN)** | 65.09% | 58.94% | 34.77% | 43.73% | 0.6433 |
| **Logistic Regression** | 49.50% | 41.60% | 52.30% | 46.30% | 0.4770 |

---

## 💻 Local Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Ostino2608/water-quality-detection-capstone.git
   cd water-quality-detection-capstone
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
