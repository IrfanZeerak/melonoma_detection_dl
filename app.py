import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2  # for image processing and heatmap overlay

# Load the trained model (ensure the .h5 file is in the same directory as this app)
model = tf.keras.models.load_model("densenet121_skin_cancer.h5")

# Class names corresponding to model output indices
class_names = ["Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma",
               "Melanoma", "Nevus", "Pigmented Benign Keratosis",
               "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"]

# App title and description
st.title("Skin Lesion Classification with Grad-CAM")
st.write("Upload a dermoscopic image of a skin lesion to predict its disease category "
         "and visualize the affected area with Grad-CAM heatmap.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image to feed into the model
    img = image.resize((224, 224))                # resize to 224x224
    img_array = np.array(img) / 255.0             # scale pixel values [0,1]
    img_tensor = np.expand_dims(img_array, axis=0)  # add batch dimension
