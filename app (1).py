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
# Load Dataset (NEW)
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Salary_Dataset_DataScienceLovers.csv")
        return df
    except Exception as e:
        st.error(f"❌ CSV loading failed: {e}")
        return None

df = load_data()

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_regressor_model_smaller.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# -------------------------------
# Dynamic Lists from CSV (NEW)
# -------------------------------
if df is not None:
    company_list = sorted(df["Company Name"].dropna().unique())
    job_title_list = sorted(df["Job Title"].dropna().unique())
    location_list = sorted(df["Location"].dropna().unique())
else:
    company_list = []
    job_title_list = []
    location_list = []

# -------------------------------
# Input Fields
# -------------------------------
rating = st.slider("⭐ Rating", 0.0, 5.0, 3.5, 0.1)

company_name = st.selectbox("🏢 Company Name", company_list)

job_title = st.selectbox("💻 Job Title", job_title_list)

salaries_reported = st.number_input("📊 Salaries Reported", min_value=1, value=5)

location = st.selectbox("📍 Location", location_list)

employment_status = st.selectbox(
    "📄 Employment Status",
    ["Full-time", "Part-time", "Intern"]
)

job_roles = st.selectbox(
    "🧩 Job Role",
    ["Backend", "Frontend", "Data", "AI"]
)

# -------------------------------
# Encoding using dataset mapping (NEW)
# -------------------------------
company_map = {name: idx for idx, name in enumerate(company_list)}
job_title_map = {name: idx for idx, name in enumerate(job_title_list)}
location_map = {name: idx for idx, name in enumerate(location_list)}

employment_map = {"Full-time": 0, "Part-time": 1, "Intern": 2}
job_roles_map = {"Backend": 0, "Frontend": 1, "Data": 2, "AI": 3}

def encode_inputs():
    return {
        "Rating": rating,
        "Company Name": company_map.get(company_name, 0),
        "Job Title": job_title_map.get(job_title, 0),
        "Salaries Reported": salaries_reported,
        "Location": location_map.get(location, 0),
        "Employment Status": employment_map.get(employment_status, 0),
        "Job Roles": job_roles_map.get(job_roles, 0),
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
