# app.py
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive file IDs
TEXT_MODEL_ID = 'https://drive.google.com/file/d/190g1lDXVk94PUUxDJETqyn5kVZoYs_5r/view?usp=drivesdk'   # Replace with actual file ID
#IMAGE_MODEL_ID = 'https://drive.google.com/file/d/1mosGYxpt4y4Yjuro62domQ9_yM2Egf47/view?usp=drivesdk'  # Replace with actual file ID

TEXT_MODEL_PATH = '/content/drive/MyDrive/btp 8 sem final/saved_model/text_model.pkl'
#IMAGE_MODEL_PATH = '/content/drive/MyDrive/btp 8 sem final/saved_model/image_model.h5'

# Download models from Google Drive if not already present
def download_models():
    if not os.path.exists(TEXT_MODEL_PATH):
        text_url = 'https://drive.google.com/file/d/190g1lDXVk94PUUxDJETqyn5kVZoYs_5r/view?usp=drivesdk'
        gdown.download(text_url, TEXT_MODEL_PATH, quiet=False)

    #if not os.path.exists(IMAGE_MODEL_PATH):
     #   image_url = 'https://drive.google.com/file/d/1mosGYxpt4y4Yjuro62domQ9_yM2Egf47/view?usp=drivesdk'
      #  gdown.download(image_url, IMAGE_MODEL_PATH, quiet=False)

# Load models
@st.cache_resource
def load_models():
    download_models()
    text_model = joblib.load(TEXT_MODEL_PATH)
    #image_model = load_model(IMAGE_MODEL_PATH)
    return text_model, #image_model

#text_model, image_model = load_models()
text_model = load_models()

# Streamlit interface
st.title("🧠 Text + Image Classifier")

# Text input
st.header("Text Input")
user_text = st.text_input("Enter your text here:")

# Image input
st.header("Image Upload")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Predict button
if st.button("Predict"):
    if user_text and uploaded_file:
        # TEXT prediction
        text_features = [user_text]  # Preprocessing can be added here if needed
        text_pred = text_model.predict(text_features)[0]

        # IMAGE prediction
        #img = Image.open(uploaded_file).resize((224, 224))
        #img_array = keras_image.img_to_array(img) / 255.0
        #img_array = np.expand_dims(img_array, axis=0)
        #image_pred = image_model.predict(img_array)
        #image_class = np.argmax(image_pred, axis=1)[0]

        # Results
        st.success(f"📝 Text Prediction: {text_pred}")
        #st.success(f"🖼️ Image Prediction Class: {image_class}")
    else:
        st.warning("Please enter text and upload an image before predicting.")

