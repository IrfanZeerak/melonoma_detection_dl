import streamlit as st
from PIL import Image
import numpy as np
import torch
from utils.inference import classify_image, segment_image
from utils.gradcam import generate_gradcam

st.set_page_config(page_title="Melanoma Detection System", layout="wide")
st.title("🧠 AI-Powered Melanoma Detection & Segmentation")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classification
    st.subheader("📊 Disease Classification")
    class_label, confidence = classify_image(image)
    st.success(f"Predicted Class: **{class_label}** ({confidence*100:.2f}% confidence)")

    # Grad-CAM Heatmap
    st.subheader("🔥 Grad-CAM Heatmap")
    gradcam_img = generate_gradcam(image)
    st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)

    # Segmentation
    st.subheader("🧬 Lesion Segmentation")
    mask = segment_image(image)
    st.image(mask, caption="U-Net Segmentation Mask", use_column_width=True)
