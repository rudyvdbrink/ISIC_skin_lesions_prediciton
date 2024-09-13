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

# %% page navigation

#create a sidebar with options for navigation
st.sidebar.title('Navigation')
st.sidebar.page_link(page="app.py", label="Home")
st.sidebar.page_link(page="pages/about.py", label="About")

# %% model definition

#model_name = 'Xception_multi-class_classifier_fully_trained_3rounds_quantized.tflite'
model_name = 'Xception_fair.tflite'

# %% functions and other definitions

#function to select a random image from the './example_img' folder
def pick_random_image(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    if image_files:
        return os.path.join(folder_path, random.choice(image_files))
    else:
        return None

class_names = ['actinic keratosis', 
                   'basal cell carcinoma', 
                   'dermatofibroma',
                   'melanoma', 
                   'nevus',
                   'pigmented benign keratosis',
                   'squamous cell carcinoma',
                   'vascular lesion']

# %% main functionality

st.title("Skin lesion prediciton") #title 

# Drag and drop box for image upload
uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])

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

# Proceed with prediction if an image (either uploaded or randomly picked) is available
if 'img' in locals():
    # Model prediction on the image
    output_data = make_tfl_prediction(model_name, img)
    probabilities = rescale_to_probability(output_data)

    predicted_class = np.argmax(output_data)
    confidence = np.round(probabilities[predicted_class])
    label = class_names[predicted_class]

    # st.write(f"Predicted class: {label}.")
    # st.write(f"Confidence: {confidence} %")

    st.markdown(f"**Predicted class: {label}.**")
    st.markdown(f"**Confidence: {confidence} %**")


    # Generate and display the prediction barplot
    fig = prediction_barplot(probabilities)
    st.pyplot(fig)



# # Drag and drop box for image upload
# uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])

# # Check if an image has been uploaded
# if uploaded_file is not None:
#     #open the image using PIL
#     img = Image.open(uploaded_file)
    
#     #display the image
#     st.image(img, caption='Uploaded Image', use_column_width=True)

#     # %% model prediction on the image

#     output_data    = make_tfl_prediction(model_name, img)
#     probabilities  = rescale_to_probability(output_data)

#     predicted_class = np.argmax(output_data)
#     confidence = probabilities[predicted_class]

#     st.write(f"Predicted class: {predicted_class}, Confidence: {confidence}")

#     fig = prediction_barplot(probabilities)
#     st.pyplot(fig)
