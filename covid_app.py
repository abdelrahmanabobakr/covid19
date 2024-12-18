import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Add background image with CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDF8fGNvcm9uYXZpcnVzfGVufDB8fHx8MTYwMjc4NzYzNw&ixlib=rb-1.2.1&q=80&w=1080");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-title {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 10px;
        border-radius: 10px;
    }
    .result-box {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description with HTML styling
st.markdown(
    """
    <div class="main-title" style="text-align: center;">
        <h1 style="color: white;">Covid-19 Classification</h1>
        <p style="color: white; font-size: 18px;">Upload an X-ray image to predict if it's Normal, Covid-19, or Viral Pneumonia.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
MODEL_PATH = "D:\\New folder\\mony\\AI projects\\ml\\covid_19_model.h5"
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Upload an image
uploaded_image = st.file_uploader(
    "Upload an X-ray Image (jpg, png, jpeg)", type=['jpg', 'png', 'jpeg']
)

# Class labels dictionary
CLASS_LABELS = {0: 'Covid', 1: 'Normal', 2: 'Viral Pneumonia'}

if uploaded_image is not None:  # Check if image is uploaded
    try:
        # Display uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Convert image to array and resize
        new_image = np.array(img)
        new_image = cv2.resize(new_image, (224, 224))

        # Normalize image
        new_image = new_image.astype('float32') / 255.0

        # Ensure image has 3 channels
        if new_image.ndim == 2:  # If grayscale, convert to RGB
            new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
        
        new_image = new_image.reshape(1, 224, 224, 3)  # Reshape to match model input

        # Prediction
        prediction = model.predict(new_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = CLASS_LABELS[predicted_class]

        # Display prediction result with styling
        st.markdown(
            f"""
            <div class="result-box" style="text-align: center;">
                <h2 style="color: #333;">Prediction Result:</h2>
                <h3 style="color: #4CAF50;">{predicted_label}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error processing image: {e}")
