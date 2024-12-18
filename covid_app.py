import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
import requests

# Title and description
st.markdown(
    """
    <div style="background-color: #4CAF50; padding: 10px; border-radius: 10px; text-align: center;">
        <h1 style="color: white;">Covid-19 Classification</h1>
        <p style="color: white; font-size: 18px;">Upload an X-ray image to predict if it's Normal, Covid-19, or Viral Pneumonia.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Google Drive model URL
url = "https://drive.google.com/uc?id=1XEGdNcngdlY0k0nT2rc3p7rYw2z7pZKP"
MODEL_PATH = "covid_19_model.h5"

# Download the model if not already downloaded
if not os.path.exists(MODEL_PATH):
    st.info("Downloading the model from Google Drive...")
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            st.success("Model downloaded successfully!")
        else:
            st.error(f"Failed to download model: {response.status_code}")
            st.stop()

# Load model
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Class labels
target_labels = {0: 'Covid', 1: 'Normal', 2: 'Viral Pneumonia'}

# File uploader
uploaded_image = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (224, 224))

    # Normalize image
    img_resized = img_resized.astype('float32') / 255.0

    # Convert grayscale to RGB (if required)
    if len(img_resized.shape) == 2:  # Grayscale
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    img_resized = img_resized.reshape(1, 224, 224, 3)

    # Predict
    predictions = model.predict(img_resized)
    predicted_class = np.argmax(predictions, axis=1)[0]
    label = target_labels[predicted_class]

    # Display result
    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #333;">Prediction Result:</h2>
            <h3 style="color: #4CAF50;">{label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
