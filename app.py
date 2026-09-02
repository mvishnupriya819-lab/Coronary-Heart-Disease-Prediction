import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("lg_CHD_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Page settings
st.set_page_config(
    page_title="Coronary Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Title
st.title("Coronary Heart Disease Prediction")

st.write("Enter Patient's Details below:")

st.markdown("---")


# ==========================
# PATIENT DETAILS
# ==========================

male = st.selectbox(
    "Gender",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=45
)

education = st.selectbox(
    "Education Level",
    [1, 2, 3, 4],
    help="1=Some high School, 2=High School/GED, 3=College, 4=Post Graduate"
)

currentSmoker = st.selectbox(
    "Current Smoker",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

cigsPerDay = st.number_input(
    "Cigarettes Per Day",
    min_value=0,
    max_value=100,
    value=0
)

BPMeds = st.selectbox(
    "BP Medication",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

prevalentStroke = st.selectbox(
    "Previous Stroke",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

prevalentHyp = st.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

diabetes = st.selectbox(
    "Diabetes",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

totChol = st.number_input(
    "Total Cholesterol",
    min_value=100,
    max_value=700,
    value=220
)

sysBP = st.number_input(
    "Systolic BP",
    min_value=70,
    max_value=300,
    value=120
)

diaBP = st.number_input(
    "Diastolic BP",
    min_value=40,
    max_value=200,
    value=80
)

BMI = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0
)

heartRate = st.number_input(
    "Heart Rate",
    min_value=30,
    max_value=200,
    value=75
)

glucose = st.number_input(
    "Glucose",
    min_value=40,
    max_value=500,
    value=80
)
# ==========================
# PREDICTION
# ==========================

if st.button("Predict"):

    # Create patient data
    input_data = pd.DataFrame([[
        male,
        age,
        education,
        currentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]], columns=[
        "male",
        "age",
        "education",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose"
    ])

    # Scale patient data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.markdown("---")

    if prediction[0] == 1:
        st.error("High Risk of Coronary Heart Disease")
    else:
        st.success("Low Risk of Coronary Heart Disease")

    st.write(
        f"**Risk Probability : {probability * 100:.2f}%**"
    )