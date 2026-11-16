import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# 2. APPLICATION TITLE
# ==========================================

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Welcome to the Telco Customer Churn Prediction System. "
    "Enter customer information below to predict whether "
    "a customer is likely to leave the company."
)

st.divider()

# ==========================================
# 3. LOAD DATASET AND TRAIN MODEL
# ==========================================

@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Remove customer ID
    df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges into numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Create Random Forest model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ]
    )

    # Train model
    model.fit(X, y)

    return model


# Load trained model
try:
    model = train_model()
    st.success("✅ Machine Learning Model Loaded Successfully!")

except FileNotFoundError:
    st.error(
        "Dataset not found! Make sure "
        "Telco-Customer-Churn.csv is in the project folder."
    )
    st.stop()

except Exception as e:
    st.error(f"Error while training model: {e}")
    st.stop()


# ==========================================
# 4. CUSTOMER INPUT FORM
# ==========================================

st.header("👤 Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    # PERSONAL INFORMATION
    with col1:

        st.subheader("Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        Partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        Dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    # SERVICE INFORMATION
    with col2:

        st.subheader("Service Details")

        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        OnlineSecurity = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        OnlineBackup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        DeviceProtection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        TechSupport = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        StreamingTV = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    # ACCOUNT INFORMATION
    st.subheader("💳 Account Details")

    col3, col4 = st.columns(2)

    with col3:

        Contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col4:

        MonthlyCharges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        TotalCharges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0
        )

    st.divider()

    # SUBMIT BUTTON
    submit = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# ==========================================
# 5. PREDICTION
# ==========================================

if submit:

    # Collect customer information
    input_data = pd.DataFrame([{

        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges

    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    st.divider()

    # ======================================
    # 6. DISPLAY PREDICTION RESULT
    # ======================================

    st.header("📈 Prediction Result")

    if prediction == "Yes":

        st.error(
            "⚠️ Customer is likely to CHURN!"
        )

        st.write(
            "This customer may leave the telecom company. "
            "The company should consider customer retention strategies."
        )

    else:

        st.success(
            "✅ Customer is likely to STAY!"
        )

        st.write(
            "This customer is likely to continue using "
            "the company's services."
        )

    # ======================================
    # 7. DISPLAY PREDICTION PROBABILITY
    # ======================================

    st.subheader("Prediction Probability")

    for label, probability in zip(
        model.classes_,
        probabilities
    ):

        if label == "Yes":
            result_label = "Churn"
        else:
            result_label = "No Churn"

        st.write(
            f"{result_label}: {probability * 100:.2f}%"
        )

        st.progress(float(probability))

    st.caption(
        "Note: Probabilities represent the model's estimated "
        "likelihood for each prediction class."
    )

# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Telco Customer Churn Prediction | "
    "Machine Learning Project | Random Forest Classifier"
)
