import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# === Download model from Google Drive ===
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/file/d/1C_iZxRwAF6NC5ZcR01jqgegXKS0YNFxJ/view?usp=sharing"  # Replace with your real file ID

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Title
st.title("🦷 Oral Disease Classification")

# Upload image
uploaded_file = st.file_uploader("Choose a mouth image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.markdown(f"### 📌 Predicted Disease: `{predicted_class}`")
