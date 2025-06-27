import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ---------- Streamlit Page Settings ----------
st.set_page_config(page_title="Insurance Charges Predictor")

# ---------- Sidebar Info (No Logo) ----------
st.sidebar.markdown("## 📝 Use Case")
st.sidebar.markdown(
    "Predict health insurance premiums using personal, demographic, and medical inputs."
)
st.sidebar.markdown("## 🔍 Model")
st.sidebar.markdown("Gradient Boosting Regressor")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("enhanced_insurance.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
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

    numerical = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
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

    return pipeline, metrics, numerical, categorical

# ---------- App Title ----------
st.title("💰 Insurance Charges Predictor")

# ---------- Load Data & Model ----------
df = load_data()
model, metrics, num_cols, cat_cols = train_model(df)
if model is None:
    st.stop()

# ---------- Input UI ----------
st.subheader("🧾 Enter Your Details")
input_data = {}
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
    'children': "Number of children/dependents",
    'smoker': "Smoker status",
    'family_history': "Major disease in family?"
}

ordered_keys = [
    'age', 'sex', 'region', 'indian_state', 'pre_existing_conditions',
    'hospital_visits_last_year', 'occupation', 'annual_income',
    'bmi', 'children', 'smoker', 'family_history'
]

for i in range(0, len(ordered_keys), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(ordered_keys):
            key = ordered_keys[i + j]
            label = key.replace("_", " ").title()
            if key in num_cols:
                val = cols[j].number_input(f"{label}", value=float(df[key].median()), help=feature_info.get(key, ""))
            else:
                val = cols[j].selectbox(f"{label}", options=df[key].dropna().unique(), help=feature_info.get(key, ""))
            input_data[key] = val

# ---------- Predict Premium ----------
if st.button("📊 Predict Premium"):
    input_df = pd.DataFrame([input_data])
    pred_log = model.predict(input_df)[0]
    predicted = np.expm1(pred_log)
    st.success(f"💸 Estimated Annual Premium: **₹ {predicted:,.2f}**")

# ---------- Show Metrics ----------
with st.expander("📈 Model Performance Metrics"):
    st.write(metrics)

