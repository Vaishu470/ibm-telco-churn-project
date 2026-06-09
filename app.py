import streamlit as st
import pandas as pd
import joblib

# Load models
models = joblib.load("telco_churn_all_models.pkl")

st.title("Telco Customer Churn Prediction")

gender = st.selectbox("Gender", [0,1])
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Partner = st.selectbox("Partner", [0,1])
Dependents = st.selectbox("Dependents", [0,1])

tenure = st.number_input("Tenure", min_value=0)

PhoneService = st.selectbox("Phone Service", [0,1])

MultipleLines = st.selectbox("Multiple Lines", [0,1,2])

InternetService = st.selectbox("Internet Service", [0,1,2])

OnlineSecurity = st.selectbox("Online Security", [0,1,2])

OnlineBackup = st.selectbox("Online Backup", [0,1,2])

DeviceProtection = st.selectbox("Device Protection", [0,1,2])

TechSupport = st.selectbox("Tech Support", [0,1,2])

StreamingTV = st.selectbox("Streaming TV", [0,1,2])

StreamingMovies = st.selectbox("Streaming Movies", [0,1,2])

Contract = st.selectbox("Contract", [0,1,2])

PaperlessBilling = st.selectbox("Paperless Billing", [0,1])

PaymentMethod = st.selectbox("Payment Method", [0,1,2,3])

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

if st.button("Predict Churn"):

    input_df = pd.DataFrame({
        'gender':[gender],
        'SeniorCitizen':[SeniorCitizen],
        'Partner':[Partner],
        'Dependents':[Dependents],
        'tenure':[tenure],
        'PhoneService':[PhoneService],
        'MultipleLines':[MultipleLines],
        'InternetService':[InternetService],
        'OnlineSecurity':[OnlineSecurity],
        'OnlineBackup':[OnlineBackup],
        'DeviceProtection':[DeviceProtection],
        'TechSupport':[TechSupport],
        'StreamingTV':[StreamingTV],
        'StreamingMovies':[StreamingMovies],
        'Contract':[Contract],
        'PaperlessBilling':[PaperlessBilling],
        'PaymentMethod':[PaymentMethod],
        'MonthlyCharges':[MonthlyCharges],
        'TotalCharges':[TotalCharges]
    })

    results = []

    for model_name, model in models.items():

        prediction = model.predict(input_df)[0]

        results.append({
            "Model": model_name,
            "Prediction": "Churn" if prediction == 1 else "No Churn"
        })

    result_df = pd.DataFrame(results)

    st.subheader("Model Predictions")
    st.dataframe(result_df)

    churn_votes = (
        result_df["Prediction"] == "Churn"
    ).sum()

    no_churn_votes = (
        result_df["Prediction"] == "No Churn"
    ).sum()

    if churn_votes > no_churn_votes:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")