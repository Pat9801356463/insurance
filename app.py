import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ---------- Page Config ----------
st.set_page_config(page_title="Insurance Charges Predictor")

# ---------- Sidebar Info ----------
st.sidebar.markdown("## 📝 Use Case")
st.sidebar.markdown("Predict health insurance premiums using personal and medical data.")
st.sidebar.markdown("## 🔍 Model")
st.sidebar.markdown("Gradient Boosting Regressor")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("enhanced_insurance.csv")
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    df.dropna(subset=['charges'], inplace=True)
    return df

# ---------- Train Model ----------
@st.cache_resource
def train_model(df):
    if df.empty:
        return None, {}, [], []

    X = df.drop(columns=['charges'])
    y = np.log1p(df['charges'])

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)),
        'RMSE': np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred))),
        'R2 Score': r2_score(np.expm1(y_test), np.expm1(y_pred))
    }

    return pipeline, metrics, num_cols, cat_cols

# ---------- Load Everything ----------
st.title("💰 Insurance Charges Predictor")
df = load_data()
model, metrics, num_cols, cat_cols = train_model(df)
if model is None:
    st.stop()

# ---------- User Input Form ----------
st.subheader("🧾 Fill in the Details")

feature_info = {
    'age': "Age of the person",
    'sex': "Gender of the insured",
    'region': "Geographical region",
    'indian_state': "State of residence",
    'pre_existing_conditions': "Any pre-existing conditions",
    'hospital_visits_last_year': "Hospital visits last year",
    'occupation': "Current occupation",
    'annual_income': "Annual income",
    'bmi': "Body Mass Index",
    'children': "Number of dependents",
    'smoker': "Smoker status",
    'family_history': "Any major disease in family?"
}

input_order = [
    'age', 'sex', 'region', 'indian_state', 'pre_existing_conditions',
    'hospital_visits_last_year', 'occupation', 'annual_income',
    'bmi', 'children', 'smoker', 'family_history'
]

user_input = {}

for i in range(0, len(input_order), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(input_order):
            key = input_order[i + j]
            label = key.replace("_", " ").title()
            if key in num_cols:
                value = cols[j].number_input(f"{label}", value=float(df[key].median()), help=feature_info[key])
            else:
                value = cols[j].selectbox(f"{label}", options=df[key].dropna().unique(), help=feature_info[key])
            user_input[key] = value

# ---------- Predict & Send to Backend ----------
if st.button("📊 Predict Premium"):
    try:
        # Predict
        input_df = pd.DataFrame([user_input])
        log_pred = model.predict(input_df)[0]
        predicted_charge = np.expm1(log_pred)
        st.success(f"💸 Estimated Annual Premium: **₹ {predicted_charge:,.2f}**")

        # Attach prediction to JSON
        payload = user_input.copy()
        payload['predicted_premium'] = round(predicted_charge, 2)

        # Send to backend
        backend_url = "https://your-backend.com/save-input"  # ⬅️ Replace with real URL
        response = requests.post(backend_url, json=payload)

        if response.status_code == 200:
            st.info("✅ Input data saved to backend successfully.")
        else:
            st.warning("⚠️ Prediction succeeded, but backend save failed.")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# ---------- Show Metrics ----------
with st.expander("📈 Model Performance Metrics"):
    st.json(metrics)
