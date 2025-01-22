# %% packages
import os
import sys
import streamlit as st
from PIL import Image
import numpy as np
import random


# %% make python able to find the functions it needs
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from supporting_functions import preprocess_image, rescale_to_probability, make_tfl_prediction, prediction_barplot

# %% config
st.set_page_config(layout="wide")

# Layout with two columns
spacer1, left_col, spacer2, right_col, spacer3 = st.columns([1, 3, 1, 3, 1])

# %% logo

st.sidebar.image('./figures/logo.png', width=250)  # Set the desired width in pixels


# %% Side bar

#create a sidebar with options for navigation
st.sidebar.title('Navigation')
st.sidebar.page_link(page="app.py", label="Home")
st.sidebar.page_link(page="pages/about.py", label="About")

#links out
st.sidebar.title('Resources')
st.sidebar.page_link(page="https://github.com/rudyvdbrink/ISIC_skin_lesions_prediciton", label="Code")
st.sidebar.page_link(page="https://ruudvandenbrink.net/", label="About author")


# %% functions and other definitions

#function to select a random image from the './example_img' folder
def pick_random_image(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    if image_files:
        return os.path.join(folder_path, random.choice(image_files))
    else:
        return None

class_names = [ 'actinic keratosis', 
                'basal cell carcinoma', 
                'dermatofibroma',
                'melanoma', 
                'nevus',
                'pigmented benign keratosis',
                'squamous cell carcinoma',
                'vascular lesion']

# %% main functionality

with left_col:

    # %% model definition

    st.title('Model selection')
    model_name = st.selectbox(
        "Select a model",   # Label for the dropdown
        ('InceptionResNet', 'Xception', 'VGG19')  # Options for the dropdown
    )

    #model_name = 'Xception_multi-class_classifier_fully_trained_3rounds_quantized.tflite'
    if model_name == 'Xception':
        model_name = 'Xception_fair.tflite'
    elif model_name == 'InceptionResNet':
        model_name = 'InceptionResNetV2_fair.tflite'
    elif model_name == 'VGG19':
        model_name = 'VGG19_fair.tflite'


    st.title("File upload") #title 

    # Drag and drop box for image upload
    uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])
    st.write("or")
    # "Pick one for me" button
    if st.button("Pick one for me"):
        random_image_path = pick_random_image('./example_imgs')
        if random_image_path:
            img = Image.open(random_image_path)
            true_class = random_image_path.split('_')[-1].split('.')[0]
            st.image(img, caption='Randomly Selected Image. True class: ' + true_class, use_column_width=True)
        else:
            st.write("No images found in the example_imgs folder.")

    # If an image has been uploaded
    elif uploaded_file is not None:
        # Open the image using PIL
        img = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

with right_col:
    # Proceed with prediction if an image (either uploaded or randomly picked) is available
    if 'img' in locals():
        # Model prediction on the image
        output_data = make_tfl_prediction(model_name, img)
        probabilities = rescale_to_probability(output_data)

        predicted_class = np.argmax(output_data)
        confidence = np.round(probabilities[predicted_class])
        label = class_names[predicted_class]

        st.title("Prediction")
        st.markdown(f"**Predicted class: {label}.**")
        st.markdown(f"**Confidence: {confidence} %**")

        # Generate and display the prediction barplot
        fig = prediction_barplot(probabilities)
        st.pyplot(fig)
