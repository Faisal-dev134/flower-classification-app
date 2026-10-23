import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

st.title("🌸 Flower Classification Model")
st.write("Flower ki photo upload karein aur model identify karega ke yeh konsa phool hai.")

# Model load karein
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model("flower_cnn.keras")

model = load_flower_model()
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# File uploader
uploaded_file = st.file_uploader("Select Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Image preprocessing
    img_resized = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Classifying Image..."):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(tf.nn.softmax(predictions[0])) * 100)

    st.success(f"**Predicted Flower:** {predicted_class.title()}")
    st.info(f"**Confidence:** {confidence:.2f}%")