# Skin lesion prediction
This repository contains code for classification of images into 8 categories of common skin lesion types, using a large convolutional neural network. Code for training the model is included. The training data consisted of both dermatoscopic and non-dermatoscopic images, and with a range of skin tones. The aim was to build a model that is less biased towards accurate diagnosis for light-skinned samples and also performs well for people of color. 

Note that no model is bias-free, as some bias will be inherent to the data on which the model was trained. The current model is only one step in the right direction, and by no means an unbiassed model.

This is not a diagnostic tool. Do not use it to diagnose your own skin lesions. This tool is intended for research purposes only. 


### Data sources

[Stanford Diverse Dermatology images](https://ddi-dataset.github.io/index.html#dataset), [publication](https://www.science.org/doi/full/10.1126/sciadv.abq6147)
\
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

### About the winning model:

Model architecture ([image source](https://www.researchgate.net/figure/Schematic-diagram-compressed-view-of-InceptionResNetv2-model_fig5_348995187)):
<img title="InceptionResNetV2" src="./figures/InceptionResNetV2_schematic.png">

Model performance on test-set:
<img title="Model Performance" src="./figures/InceptionResNetV2_performance.png">


### **Installation on `macOS`**: 


- Install the virtual environment and the required packages:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **Installation on `WindowsOS`**:

- Install the virtual environment and the required packages:

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :

  ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```