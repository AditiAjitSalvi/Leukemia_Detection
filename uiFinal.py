# v5_ui.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import joblib

# Import deep learning parts
from V5 import SwinTransformer, GraphNeuralNetwork, predict_image, load_models

# ---- Page Configuration ----
st.set_page_config(page_title="Leukemia Detection System", page_icon="🧬", layout="wide")

# ---- Header Section ----
st.title("🧫 Leukemia Detection System")
st.markdown("""
This AI-based system detects the **type and stage of leukemia** using:
- 🧪 **CBC / Bone Marrow Data (SVM Model)**
- 🧬 **WBC Image (Swin Transformer + GNN Model)**
""")

# ---- Load DL Models ----
@st.cache_resource
def load_trained_models():
    if not os.path.exists('swin_transformer_model.pth') or not os.path.exists('gnn_stage_model.pth'):
        st.error("❌ Deep learning models not found. Please train using V5.py.")
        st.stop()
    return load_models()

swin_model, gnn_model = load_trained_models()

# ---- Sidebar: Image Upload ----
st.sidebar.header("📤 Upload WBC Image")
uploaded_file = st.sidebar.file_uploader("Upload WBC Image", type=["jpg", "jpeg", "png"])

# ---- WBC Image-Based Prediction ----
st.header("🔬 WBC Image-Based Leukemia Stage Detection")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded WBC Image", use_container_width=True)

    if st.button("🧠 Predict Leukemia Stage"):
        temp_path = "temp_uploaded_image.jpg"
        image.save(temp_path)
        try:
            prediction = predict_image(temp_path, swin_model, gnn_model)
            st.success(f"🧠 Predicted Leukemia Stage: **{prediction}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("📤 Please upload a WBC image to start.")

# ---- CBC / Bone Marrow Based Prediction ----
st.header("🧪 CBC / Bone Marrow Data - Leukemia Type Detection")

# ✅ Load SVM Model, Scaler, Encoder
try:
    svm_model = joblib.load("svm_leukemia_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
except:
    st.error("❌ Could not load SVM model or scaler. Please ensure 'svm_leukemia_model.pkl', 'scaler.pkl', and 'encoder.pkl' are available.")
    st.stop()

# ✅ Extract training column names automatically (ensures same structure)
columns = list(joblib.load("scaler.pkl").mean_.shape)
# But mean_.shape only gives count, not names, so let's use:
# Load feature names directly from a saved CSV (recommended)
# OR define manually exactly as in training:
columns = [
    'WBC', 'RBC', 'Hemoglobin', 'Platelet',
    'Myeloblast%', 'Lymphoblast%', 'M:E Ratio', 'LDH', 'Age'
]

with st.form("cbc_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        wbc = st.number_input("WBC count (cells/µL)", min_value=0.0)
        rbc = st.number_input("RBC count (million/µL)", min_value=0.0)
        hb = st.number_input("Hemoglobin (g/dL)", min_value=0.0)
    with col2:
        platelet = st.number_input("Platelet count (×10³/µL)", min_value=0.0)
        myeloblast = st.number_input("Myeloblast %", min_value=0.0)
        lymphoblast = st.number_input("Lymphoblast %", min_value=0.0)
    with col3:
        me_ratio = st.number_input("M:E Ratio", min_value=0.0)
        ldh = st.number_input("LDH (U/L)", min_value=0.0)
        age = st.number_input("Age (years)", min_value=0.0)

    submitted = st.form_submit_button("🩸 Predict Leukemia Type")

if submitted:
    try:
        # Prepare data frame with correct feature names
        features = np.array([[wbc, rbc, hb, platelet, myeloblast, lymphoblast, me_ratio, ldh, age]])
        features_df = pd.DataFrame(features, columns=columns)

        # Scale input
        features_scaled = scaler.transform(features_df)

        # Predict leukemia type
        pred = svm_model.predict(features_scaled)
        leukemia_type = encoder.inverse_transform(pred)[0]

        # Display result
        st.success(f"🧫 Predicted Leukemia Type: **{leukemia_type}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---- Footer ----
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Developed by Aditi and Chetna </p>", unsafe_allow_html=True)
