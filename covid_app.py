import requests
import os
from tensorflow.keras.models import load_model
import streamlit as st

# رابط Google Drive
FILE_ID = "1XEGdNcngdlY0k0nT2rc3p7rYw2z7pZKP"
url = f"https://drive.google.com/uc?id={FILE_ID}"

# مسار النموذج المحلي
MODEL_PATH = "covid_19_model.h5"

# تحميل النموذج ديناميكيًا
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

# تحميل النموذج
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()
