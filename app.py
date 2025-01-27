# %% packages
import os
import sys
import streamlit as st
from PIL import Image
import numpy as np

# %% make python able to find the functions it needs
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from supporting_functions import (draw_st_bar_chart, 
                                  rescale_to_probability,
                                  make_tfl_prediction, 
                                  pick_random_image
                                 )

from shap_functions import compute_shap_values, plot_shap_values


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

# %% class definition

class_names = [ 'actinic keratosis', 
                'basal cell carcinoma', 
                'dermatofibroma',
                'melanoma', 
                'nevus',
                'pigmented benign keratosis',
                'squamous cell carcinoma',
                'vascular lesion']

# %% Initialize session state variables

if 'img' not in st.session_state:
    st.session_state.img = None
if 'random_pick' not in st.session_state:
    st.session_state.random_pick = False
if 'true_class' not in st.session_state:
    st.session_state.true_class = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'class_to_show' not in st.session_state:
    st.session_state.class_to_show = None

# %% main functionality

with left_col:

    # %% model definition
    st.title('Model selection')
    with st.container(border=True):
       
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
    with st.container(border=True):
        
        # Drag and drop box for image upload
        uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])
        st.write("or")
        # "Pick one for me" button
        if st.button("Pick one for me",key='random_img'):
            random_image_path = pick_random_image('./example_imgs')
            if random_image_path:
                st.session_state.random_pick = True
                st.session_state.shap_values = None
                st.session_state.img = Image.open(random_image_path) # Open the image using PIL
                st.session_state.true_class = random_image_path.split('_')[-1].split('.')[0] # Get true class
            else:
                st.write("No images found in the example_imgs folder.")

        # If an image has been uploaded
        elif uploaded_file is not None:
            
            st.session_state.random_pick = False
            st.session_state.img = Image.open(uploaded_file) # Open the image using PIL

        # Display the image
        if st.session_state.img is not None:
            if st.session_state.random_pick == True:
                st.image(st.session_state.img, caption='Randomly Selected Image. True class: ' + st.session_state.true_class, use_column_width=True)              
            else:
                st.image(st.session_state.img, caption='Uploaded Image', use_column_width=True)


with right_col:
    # Proceed with prediction if an image (either uploaded or randomly picked) is available
    if st.session_state.img is not None:
        # Model prediction on the image
        output_data = make_tfl_prediction(model_name, st.session_state.img)
        probabilities = rescale_to_probability(output_data)

        #print(probabilities)

        predicted_class = np.argmax(output_data)
        confidence = int(np.round(probabilities[predicted_class]))
        label = class_names[predicted_class]

        st.title("Prediction")
        with st.container(border=True):

            # Descriptive text of model prediction
            st.markdown(f"The model is **{confidence}%** sure that this is a **{label}**.")

            # Generate and display the prediction barplot
            draw_st_bar_chart(probabilities, class_names)

            with st.popover("What do the labels mean?"):
                   st.markdown("""
                               ### Diagnostic categories

                                *Actinic keratosis* is a rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous, as it can potentially evolve into squamous cell carcinoma if left untreated.

                                *Basal cell carcinoma* is the most common type of skin cancer, typically caused by long-term sun exposure. It grows slowly and rarely spreads to other parts of the body, but early treatment is recommended to prevent local tissue damage.

                                *Dermatofibroma* is a benign skin growth, usually firm and raised, often resulting from a minor injury like a bug bite. It’s generally harmless and does not require treatment unless it becomes symptomatic or bothersome.

                                *Melanoma* is a highly aggressive form of skin cancer that arises from melanocytes, the cells responsible for skin pigmentation. It has a high potential to spread to other organs, making early detection and treatment critical for a good prognosis.

                                *Nevus*, or mole, is a common benign skin lesion caused by clusters of melanocytes. While most nevi are harmless, some may undergo changes that necessitate monitoring for melanoma risk.

                                *Pigmented benign keratosis*, such as seborrheic keratosis, is a common non-cancerous skin growth that appears as a brown, black, or pale patch. These lesions are typically harmless and do not require treatment unless for cosmetic reasons or irritation.

                                *Squamous cell carcinoma*, is a type of skin cancer that develops from the squamous cells of the epidermis. It can grow and spread if untreated, but it is generally curable when caught early.

                                *Vascular lesion* is an abnormality of the skin or mucous membranes caused by blood vessels, including conditions like hemangiomas and cherry angiomas. Most vascular lesions are benign and often do not require treatment unless for cosmetic purposes or if symptomatic.

                                """)
            
        st.title("Detail analysis")
        with st.container(border=True):
            if st.button("Analyze further", key="analyze"):
                
                with st.spinner('Analyzing image...'):
                    st.session_state.shap_values = compute_shap_values(model_name, st.session_state.img)
                    st.session_state.class_to_show = predicted_class  

            if st.session_state.shap_values is not None:     

                if st.session_state.class_to_show == None:
                    st.session_state.class_to_show = predicted_class           
            
                fig, ax = plot_shap_values(st.session_state.shap_values, st.session_state.img, st.session_state.class_to_show)                
                st.pyplot(fig)
                
                class_to_show = st.selectbox("Show values for:", options=class_names, index=int(st.session_state.class_to_show))
                st.session_state.class_to_show = class_names.index(class_to_show)

                with st.popover("What does this mean?"):
                    st.markdown("The SHAP values indicate the contribution of each pixel to the model's prediction. Positive values (in red) indicate that the pixel contributes to the model's prediction, while negative values (in blue) indicate that the pixel contradicts the model's prediction. The intensity of the color indicates the strength of the contribution. \n\n\n This can help determine if the model is paying attention to the right parts of the image. For example, if it is using the lesion itself to determine the class rather than the background.")
