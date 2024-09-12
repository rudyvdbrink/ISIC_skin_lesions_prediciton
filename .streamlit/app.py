# %% packages
import os
import sys
import streamlit as st
from PIL import Image
import numpy as np


# %% make python able to find the functions it needs
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from supporting_functions import preprocess_image, rescale_to_probability, make_tfl_prediction, prediction_barplot

# %% formatting

# # Load and inject the CSS file
# with open("styles.css") as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# %% model definition

#model_name = 'Xception_multi-class_classifier_fully_trained_3rounds_quantized.tflite'
model_name = 'Xception_fair.tflite'

# %%

st.title("Skin lesion prediciton") #title 

# Drag and drop box for image upload
uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])

# Check if an image has been uploaded
if uploaded_file is not None:
    #open the image using PIL
    img = Image.open(uploaded_file)
    
    #display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Get image dimensions
    #width, height = img.size
    
    # Display the dimensions
    #st.write(f"Dimensions of the image: {width} x {height} pixels")

    # %% model prediction on the image

    output_data    = make_tfl_prediction(model_name, img)
    probabilities  = rescale_to_probability(output_data)

    predicted_class = np.argmax(output_data)
    confidence = probabilities[predicted_class]

    st.write(f"Predicted class: {predicted_class}, Confidence: {confidence}")


    fig = prediction_barplot(probabilities)
    st.pyplot(fig)
