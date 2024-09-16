# %% packages

import streamlit as st
from PIL import Image

# %%

st.set_page_config(layout="wide")

st.title("Skin lesion prediction")

# %% page navigation
# Create a sidebar with options for navigation
st.sidebar.title('Navigation')

st.sidebar.page_link(page="app.py", label="Home")
st.sidebar.page_link(page="pages/about.py", label="About")

# Play audio in the sidebar
st.sidebar.title('Audio summary')
audio_file = open('.streamlit/audio_summary.wav', 'rb')
audio_bytes = audio_file.read()
st.sidebar.audio(audio_bytes, format='audio/wav')
st.sidebar.write("Made with notebookLM")

#drop-down menu to select a model
st.sidebar.title('Model selection')
model_name = st.sidebar.selectbox(
    "Select a model",   # Label for the dropdown
    ('InceptionResNet', 'Xception')  # Options for the dropdown
)

# %% Layout with two columns
left_col, right_col = st.columns(2)

# %% content

# Left column: General information
with left_col:
    st.markdown("""

    ### Aim and scope

    This tool performs classification of images into 8 categories of common skin lesion types, using a large convolutional neural network. The training data consisted of both dermatoscopic and non-dermatoscopic images, and with a range of skin tones. The aim was to build a model that is less biased towards accurate diagnosis for light-skinned samples and also performs well for people with darker complexions. 

    Note that no model is bias-free, as some bias will be inherent to the data on which the model was trained. The current model is only one step in the right direction, and by no means an unbiassed model.

    This is not a diagnostic tool. Do not use it to diagnose your own skin lesions. This tool is intended to aid rapid intial screening by medical professionals. Always consult a medical professional.   


    ### Data sources

    [Stanford Diverse Dermatology images](https://ddi-dataset.github.io/index.html#dataset), [publication](https://www.science.org/doi/full/10.1126/sciadv.abq6147)


    [ISIC 2019 challenge](https://api.isic-archive.com/collections/65/), [description](https://challenge.isic-archive.com/landing/2019/)\


    [HAM10000](https://api.isic-archive.com/collections/212/), [publication](https://www.nature.com/articles/sdata2018161#MOESM246)

    ### Diagnosis categories

    `Actinic keratosis (AK)` is a rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous, as it can potentially evolve into squamous cell carcinoma if left untreated.

    `Basal cell carcinoma (BCC)` is the most common type of skin cancer, typically caused by long-term sun exposure. It grows slowly and rarely spreads to other parts of the body, but early treatment is recommended to prevent local tissue damage.

    `Dermatofibroma (DF)` is a benign skin growth, usually firm and raised, often resulting from a minor injury like a bug bite. It’s generally harmless and does not require treatment unless it becomes symptomatic or bothersome.

    `Melanoma (MLN)` is a highly aggressive form of skin cancer that arises from melanocytes, the cells responsible for skin pigmentation. It has a high potential to spread to other organs, making early detection and treatment critical for a good prognosis.

    `Nevus (NV)`, or mole, is a common benign skin lesion caused by clusters of melanocytes. While most nevi are harmless, some may undergo changes that necessitate monitoring for melanoma risk.

    `Pigmented benign keratosis (PBK)`, such as seborrheic keratosis, is a common non-cancerous skin growth that appears as a brown, black, or pale patch. These lesions are typically harmless and do not require treatment unless for cosmetic reasons or irritation.

    `Squamous cell carcinoma (SCC)`, is a type of skin cancer that develops from the squamous cells of the epidermis. It can grow and spread if untreated, but it is generally curable when caught early.

    `Vascular lesion (VL)` is an abnormality of the skin or mucous membranes caused by blood vessels, including conditions like hemangiomas and cherry angiomas. Most vascular lesions are benign and often do not require treatment unless for cosmetic purposes or if symptomatic.


    """)

# Right column: Model information
if model_name == 'InceptionResNet':
    with right_col:
        st.markdown(
        """    
        ### About the model

        InceptionResNetV2 is a deep convolutional neural network architecture that combines the strengths of the Inception architecture and residual connections from ResNet. The model was introduced by [Szegedy et al.](https://dl.acm.org/doi/10.5555/3298023.3298188) and builds on the Inception modules, which allow for multiple filter sizes to operate on the same level, thereby capturing features at different scales. The addition of residual connections, inspired by the ResNet architecture, helps to mitigate the vanishing gradient problem and makes training faster and more efficient for deeper networks. This hybrid design offers a good balance between computational complexity and accuracy.
        """
        )
        st.markdown("Model architecture ([image source](https://www.researchgate.net/figure/Schematic-diagram-compressed-view-of-InceptionResNetv2-model_fig5_348995187)):")
        #img = Image.open("./figures/InceptionResNetV2_schematic.png")
        img = Image.open("./figures/InceptionResNetV2_schematic.png")

        st.image(img, caption='InceptionResNetV2 architecture', use_column_width=True)

        st.markdown("Model performance on test-set:")
        img = Image.open("./figures/InceptionResNetV2_performance.png")
        st.image(img, caption='Model performance', use_column_width=True)
elif model_name == 'Xception':
    with right_col:        
        st.markdown(
        """    
        ### About the model

        The Xception model, introduced by [Chollet](https://ieeexplore.ieee.org/document/8099678) in 2017, is a deep convolutional neural network architecture that improves upon the Inception model by replacing the traditional Inception modules with depthwise separable convolutions. This results in a more efficient use of model parameters and computational resources. Xception, which stands for "Extreme Inception," leverages the idea that spatial and depthwise feature learning can be decoupled, allowing for the use of simpler convolutional operations that maintain high performance."""
        )
        st.markdown("Model architecture ([image source](https://www.researchgate.net/figure/Schematic-diagram-of-the-Xception-model_fig3_352247462)):")
        #img = Image.open("./figures/InceptionResNetV2_schematic.png")
        img = Image.open("./figures/Xception_schematic.png")

        st.image(img, caption='Xception architecture', use_column_width=True)

        st.markdown("Model performance on test-set:")
        img = Image.open("./figures/Xception_performance.png")
        st.image(img, caption='Model performance', use_column_width=True)
