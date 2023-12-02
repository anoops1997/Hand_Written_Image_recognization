import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image  # Import Image module from PIL for image processing

# Load the trained model
model = load_model('mnist_model.h5')

# Streamlit app
st.title("MNIST Digit Classification App")
st.sidebar.header("User Input")

# Upload image through streamlit sidebar
uploaded_file = st.sidebar.file_uploader("Choose a digit image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.sidebar.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image for the model
    img = Image.open(uploaded_file)  # Use 'img' instead of 'image'
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1)).astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    st.write("")
    st.write("Classifying...")
    
    # Display the predicted digit and confidence
    st.write(f"Predicted Digit: {predicted_class}")
    st.write(f"Confidence: {prediction[0][predicted_class]:.2%}")
