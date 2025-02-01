# %% packages

import streamlit as st
from PIL import Image

# %% Layout with one column

#set page layout to normal
st.set_page_config(layout="centered")
st.title("Skin Lesion Identifier")

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


# %% content

# Left column: General information
st.markdown("""

### Aim and scope

This app classifies spots on the skin into 8 categories (some cancerous, some otherwise) using a large convolutional neural network. The aim was to build a model that performs well for everyone, including people of color. 

Note that no model is bias-free, as some bias will be inherent to the data on which the model was trained. The current model is only one step in the right direction, and by no means an unbiassed model.

This is not a diagnostic tool. Do not use it to diagnose your own skin lesions. With that in mind, feel free to try out the app on the home page. You can upload your own image, or simply use one of the example images provided.             
"""            
)


st.markdown("""
### Additional information

            """)

with st.expander("About the data"):
    st.markdown("""
    ### Data sources
    The training data consisted of both dermatoscopic and non-dermatoscopic images, and with a range of skin tones. The following sources were used:

    [Stanford Diverse Dermatology images](https://ddi-dataset.github.io/index.html#dataset), [publication](https://www.science.org/doi/full/10.1126/sciadv.abq6147)


    [ISIC 2019 challenge](https://api.isic-archive.com/collections/65/), [description](https://challenge.isic-archive.com/landing/2019/)\


    [HAM10000](https://api.isic-archive.com/collections/212/), [publication](https://www.nature.com/articles/sdata2018161)
    """
    )

with st.expander("Diagnostic categories"):
    st.markdown("""
        ### Diagnostic categories

        `Actinic keratosis (AK)` is a rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous, as it can potentially evolve into squamous cell carcinoma if left untreated.

        `Basal cell carcinoma (BCC)` is the most common type of skin cancer, typically caused by long-term sun exposure. It grows slowly and rarely spreads to other parts of the body, but early treatment is recommended to prevent local tissue damage.

        `Dermatofibroma (DF)` is a benign skin growth, usually firm and raised, often resulting from a minor injury like a bug bite. It’s generally harmless and does not require treatment unless it becomes symptomatic or bothersome.

        `Melanoma (MLN)` is a highly aggressive form of skin cancer that arises from melanocytes, the cells responsible for skin pigmentation. It has a high potential to spread to other organs, making early detection and treatment critical for a good prognosis.

        `Nevus (NV)`, or mole, is a common benign skin lesion caused by clusters of melanocytes. While most nevi are harmless, some may undergo changes that necessitate monitoring for melanoma risk.

        `Pigmented benign keratosis (PBK)`, such as seborrheic keratosis, is a common non-cancerous skin growth that appears as a brown, black, or pale patch. These lesions are typically harmless and do not require treatment unless for cosmetic reasons or irritation.

        `Squamous cell carcinoma (SCC)`, is a type of skin cancer that develops from the squamous cells of the epidermis. It can grow and spread if untreated, but it is generally curable when caught early.

        `Vascular lesion (VL)` is an abnormality of the skin or mucous membranes caused by blood vessels, including conditions like hemangiomas and cherry angiomas. Most vascular lesions are benign and often do not require treatment unless for cosmetic purposes or if symptomatic.


        """)

with st.expander("Model technical information"):
    model_name = st.selectbox(
        "Select a model",   # Label for the dropdown
        ('InceptionResNet', 'Xception', 'VGG19'))  # Options for the dropdown
    
    # Right column: Model information
    if model_name == 'InceptionResNet':
        
        
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
    elif model_name == 'VGG19':
        
        st.markdown(
        """    
        ### About the model

        The VGG19 model is a deep convolutional neural network introduced by the Visual Geometry Group (VGG) at Oxford in 2014. It is an extension of the VGG16 architecture and consists of 19 layers, including 16 convolutional layers followed by 3 fully connected layers. VGG19 uses small 3x3 convolution filters applied repeatedly in deep stacks. 
        """
        )
        st.markdown("Model architecture ([image source](https://www.researchgate.net/figure/Schematic-diagram-of-the-Xception-model_fig3_352247462)):")
        img = Image.open("./figures/VGG19_schematic.png")

        st.image(img, caption='VGG19 architecture', use_column_width=True)

        st.markdown("Model performance on test-set:")
        img = Image.open("./figures/VGG19_performance.png")
        st.image(img, caption='Model performance', use_column_width=True)        


# %% audio summary

st.markdown("""
### Audio summary
""")

audio_file = open('.streamlit/audio_summary.wav', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/wav')