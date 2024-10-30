import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Loading the model
model = load_model("animal_classifier.h5")


class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
               'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Streamlit app structure
st.title("Animal Image Classifier")
st.write("Upload an image to classify it into one of the animal classes.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Preprocessing the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    score = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    #st.write(f"Prediction: {predicted_class} with confidence {score:.2f}")

    confidence_threshold = 0.6
    # Displaying results
    if score >= confidence_threshold:

        st.write(f'Predicted class: {predicted_class} with confidence: {score:.2f}')
    else:
        st.error("The model is not confident in its prediction. Please try another image.")