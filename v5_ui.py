import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import os

# Import your existing code (SwinTransformer, GraphNeuralNetwork, predict_image)
from V5 import SwinTransformer, GraphNeuralNetwork, predict_image, load_models

# ---- Page Configuration ----
st.set_page_config(page_title="Leukemia Detection System", page_icon="🧬", layout="wide")

# ---- Header Section ----
st.title("🧫 Leukemia Detection System")
st.markdown("""
This AI-based tool helps detect the **stage of leukemia** from white blood cell images using a hybrid **Swin Transformer + Graph Neural Network** model.
""")

# ---- Load Model (Only Once) ----
@st.cache_resource
def load_trained_models():
    if not os.path.exists('swin_transformer_model.pth') or not os.path.exists('gnn_stage_model.pth'):
        st.error("❌ Model weights not found. Please train the model first by running V5.py.")
        st.stop()
    swin, gnn = load_models()
    return swin, gnn

swin_model, gnn_model = load_trained_models()

# ---- Sidebar ----
st.sidebar.header("🔍 Upload Cell Image")
uploaded_file = st.sidebar.file_uploader("Upload a WBC Image", type=["jpg", "jpeg", "png"])

# ---- Prediction Section ----
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded WBC Image", use_container_width=True)

    # Add a button for prediction
    if st.button("🔬 Predict Leukemia Stage"):
        # Save temp image for prediction
        temp_path = "temp_uploaded_image.jpg"
        image.save(temp_path)

        # Run prediction
        try:
            prediction = predict_image(temp_path, swin_model, gnn_model)
            st.success(f"🧠 Predicted Leukemia Stage: **{prediction}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("📤 Please upload an image from the sidebar to begin.")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by Aditi and Chetna </p>",
    unsafe_allow_html=True
)

#python -m streamlit run v5_ui.py
