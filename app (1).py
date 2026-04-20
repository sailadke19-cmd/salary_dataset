import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💼")

st.title("💼 Salary Prediction App")
st.write("Enter details below to predict salary")

# -------------------------------
# Debug: Check files
# -------------------------------
if st.checkbox("Show files in directory"):
    st.write(os.listdir())

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# -------------------------------
# Input Fields
# -------------------------------
rating = st.slider("⭐ Rating", 0.0, 5.0, 3.5, 0.1)

company_name = st.selectbox(
    "🏢 Company Name",
    ["TCS", "Infosys", "Wipro", "Google", "Amazon"]
)

job_title = st.selectbox(
    "💻 Job Title",
    ["Data Scientist", "Software Engineer", "Analyst", "ML Engineer"]
)

salaries_reported = st.number_input("📊 Salaries Reported", min_value=1, value=5)

location = st.selectbox(
    "📍 Location",
    ["Mumbai", "Bangalore", "Delhi", "Hyderabad"]
)

employment_status = st.selectbox(
    "📄 Employment Status",
    ["Full-time", "Part-time", "Intern"]
)

job_roles = st.selectbox(
    "🧩 Job Role",
    ["Backend", "Frontend", "Data", "AI"]
)

# -------------------------------
# Simple Encoding (MUST match training)
# -------------------------------
def encode_inputs():
    return {
        "Rating": rating,
        "Company Name": hash(company_name) % 1000,
        "Job Title": hash(job_title) % 1000,
        "Salaries Reported": salaries_reported,
        "Location": hash(location) % 100,
        "Employment Status": hash(employment_status) % 10,
        "Job Roles": hash(job_roles) % 10,
    }

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Salary"):

    if model is None:
        st.error("Model not loaded. Check model.pkl")
    else:
        try:
            input_dict = encode_inputs()

            input_df = pd.DataFrame([input_dict])

            prediction = model.predict(input_df)[0]

            st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
