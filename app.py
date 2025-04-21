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


    # Make prediction
    predictions = model.predict(img_tensor)
    pred_class_idx = np.argmax(predictions[0])
    pred_class_name = class_names[pred_class_idx]
    
    # Display the prediction result
    st.markdown(f"**Predicted Disease:** {pred_class_name}")
    # Grad-CAM: get the last conv layer's output and gradient with respect to the predicted class
    # Identify the last convolutional layer in DenseNet121
    last_conv_layer = model.get_layer('conv5_block16_concat')
    grad_model = tf.keras.models.Model(inputs=model.inputs,
                                      outputs=[last_conv_layer.output, model.output])
    # Compute gradients for the predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, pred_class_idx]       # score for the predicted class
    # Gradient of the loss with respect to conv layer output
    grads = tape.gradient(loss, conv_outputs)[0]    # grads shape: (H, W, channels)
    conv_outputs = conv_outputs[0]                  # conv feature map
    
    # Global average pooling of gradients to get weights
    weights = tf.reduce_mean(grads, axis=(0, 1))    # shape: (channels,)
    
    # Compute weighted combination of feature maps
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    
    # Apply ReLU to discard negative influence, then normalize heatmap to [0,1]
    cam = np.maximum(cam, 0)
    heatmap = cam / (np.max(cam) + 1e-8)
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap.numpy(), (image.width, image.height))
    heatmap = (heatmap * 255).astype("uint8")        # scale to [0,255] as uint8
    
    # Apply a colormap (JET) to the heatmap for visualization
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on the original image
    orig_img = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    
    # Display the Grad-CAM heatmap
    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

