# Import Required Libraries

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit Page Configuration

st.set_page_config(page_title="💰 Loan Predictor", layout="wide")
st.title("💰 Loan Approval Classifier")
st.caption("This ML project predicts loan approval status using a sample dataset. For practice purposes only.")


# Function to Load CSV Data

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


# Function to Train Model

@st.cache_resource
def train_model(df: pd.DataFrame):
    target_col = "approved"
    exclude_cols = [target_col]

    if "applicant_name" in df.columns:
        exclude_cols.append("applicant_name")

    X = df.drop(columns=exclude_cols)
    y = df[target_col]

    # Identify categorical and numerical features
    cat_features = [c for c in ["gender", "city", "employment_type", "bank"] if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # Logistic Regression Model
    model = LogisticRegression(max_iter=2000)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    metrics_dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return pipeline, metrics_dict, X_train.columns.tolist()

# Sidebar: Load Dataset

st.sidebar.header("Step 1: Load Dataset")

csv_file = st.sidebar.text_input(
    "CSV File Path",
    value="loan_dataset.csv",
    help="Put the path to the CSV file (default if in same folder)."
)

try:
    df = load_data(csv_file)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows successfully")

# Sidebar: Train Model

st.sidebar.header("Step 2: Train Model")
train_btn = st.sidebar.button("Train / Re-train")

if train_btn:
    st.cache_resource.clear()

model, metrics, feature_order = train_model(df)

# Main Layout

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Model Performance")
    st.write({
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1 Score": round(metrics["f1"], 4),
    })

    cm = np.array(metrics["conf_matrix"])
    st.write("Confusion Matrix (rows=actual, columns=predicted)")
    st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]), use_container_width=True)

st.markdown("---")

# Prediction Inputs

st.subheader("Try a Prediction")
c1, c2, c3, c4 = st.columns(4)

with c1:
    applicant_name = st.text_input("Applicant Name", value="Ali Khan")
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.slider("Age", 20, 65, 30)

with c2:
    city = st.selectbox("City", sorted(df["city"].unique()))
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique()))
    bank = st.selectbox("Bank", sorted(df["bank"].unique()))

with c3:
    monthly_income = st.number_input("Monthly Income (PKR)", min_value=1000, max_value=1000000, value=100000, step=5000)
    credit_score = st.slider("Credit Score", 300, 900, 650)

with c4:
    loan_amount = st.number_input("Loan Amount (PKR)", min_value=50000, max_value=5000000, value=750000, step=5000)
    loan_tenure = st.selectbox("Tenure (months)", [6,12,24,36,48,60])
    existing_loans = st.selectbox("Existing Loans", [0,1,2,3])
    default_history = st.selectbox("Default History", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    has_credit_card = st.selectbox("Has Credit Card?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

# Build Input Row & Predict

input_data = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "city": city,
    "employment_type": employment_type,
    "bank": bank,
    "monthly_income_pkr": monthly_income,
    "credit_score": credit_score,
    "loan_amount_pkr": loan_amount,
    "loan_tenure_months": loan_tenure,
    "existing_loans": existing_loans,
    "default_history": default_history,
    "has_credit_card": has_credit_card
}])

# Match columns
input_data = input_data[feature_order]

if st.button("Predict Approval"):
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_data)[:,1][0])
    else:
        prob = float(model.predict(input_data)[0])

    prediction = int(prob >= 0.5)

    if prediction == 1:
        st.success(f"{applicant_name} : APPROVED ✅ (Probability: {prob:.2%})")
    else:
        st.error(f"{applicant_name} : REJECTED ❌ (Probability: {prob:.2%})")
