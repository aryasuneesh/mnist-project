import streamlit as st
import requests
import numpy as np
from PIL import Image

# Title and instructions
st.title('Handwritten Digit Recognition')
st.write('Upload an image of a handwritten digit and click the Predict button to see the model prediction.')

@st.cache(suppress_st_warning=True)
def predict_digit(image):
    # Send image data to Flask API for prediction
    response = requests.post('http://localhost:5000/predict', json={'data': image.tolist()})
    return response.json()['prediction'][0]

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Predict button
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to grayscale and resize to match model input size
    image_array = np.array(image.convert('L').resize((28, 28)))

    # Flatten the image array and reshape it to match model input shape
    image_flattened = image_array.flatten().reshape(1, -1)

    # Normalize pixel values
    image_normalized = image_flattened / 255.0

    # Call the predict_digit function to get prediction
    prediction = predict_digit(image_normalized)
    
    # Display prediction
    st.write(f'Prediction: {prediction}')
