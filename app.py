
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "loan_default_model (2).pkl"

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Unable to load ML model: {e}")
    st.stop()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stButton button {
        width: 100%;
        height: 45px;
        font-size: 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🏦 Bank Loan Default Risk Predictor")

st.write(
    "AI-powered loan risk assessment system"
)

st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=70,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    employment = st.selectbox(
        "Employment Type",
        [
            "Salaried",
            "Self Employed",
            "Business"
        ]
    )

with col2:

    income = st.number_input(
        "Annual Income",
        min_value=100000,
        max_value=10000000,
        value=500000
    )

    credit = st.slider(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    loan_type = st.selectbox(
        "Loan Type",
        [
            "Home Loan",
            "Personal Loan",
            "Auto Loan",
            "Education Loan"
        ]
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=100000,
        max_value=50000000,
        value=500000
    )

with col3:

    interest = st.number_input(
        "Interest Rate",
        min_value=5.0,
        max_value=20.0,
        value=10.0,
        step=0.5
    )

    debt = st.slider(
        "Debt To Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01
    )

    collateral = st.number_input(
        "Collateral Value",
        min_value=0,
        max_value=100000000,
        value=0
    )

    existing = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=10,
        value=0
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Risk"):

    if income <= 0:
        st.error("Annual income must be greater than zero.")

    else:

        try:

            input_data = pd.DataFrame({

                "Age": [age],

                "Gender": [gender],

                "Marital_Status": ["Married"],

                "Dependents": [1],

                "Education": [education],

                "Employment_Type": [employment],

                "Annual_Income": [income],

                "Monthly_Income": [income / 12],

                "Credit_Score": [credit],

                "Existing_Loan_Count": [existing],

                "Loan_Type": [loan_type],

                "Loan_Amount": [loan_amount],

                "Loan_Term": [60],

                "Interest_Rate": [interest],

                "Debt_to_Income_Ratio": [debt],

                "Collateral_Value": [collateral],

                "Loan_Region": ["East"],

                "Disbursement_Status": ["Disbursed"],

                "Repayment_Status": ["On Time"]

            })

            # Run model prediction
            result = model.predict(input_data)

            st.subheader("Prediction Result")

            if result[0] == 1:

                st.error("⚠️ High Default Risk")

                st.warning(
                    "The model predicts a higher likelihood of loan default."
                )

            else:

                st.success("✅ Low Default Risk")

                st.info(
                    "The model predicts a lower likelihood of loan default."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )
