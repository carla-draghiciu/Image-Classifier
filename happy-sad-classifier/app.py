import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("C:/Users/dragh/anaconda3/Scripts/Image-Classification/models/happysadmodel.keras")

st.title("Emotion Classifier")
st.write("Upload an image, and the model will predict if it’s happy or sad!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # from (256, 256, 3) to (1, 256, 256, 3) 1=batch size

    prediction = model.predict(img_array)
    label = "Happy" if prediction < 0.5 else "Sad"

    st.subheader(f"Prediction: {label}")
