# ISIC_skin_lesions_prediciton
Prediction of malignance of skin lesions using data from the International Skin Imaging Collaboration​

Data source: [HAM10000](https://api.isic-archive.com/collections/212/)

Credit: https://www.nature.com/articles/sdata2018161#MOESM246

### List of files:
- `0_xyz.ipynb`: xyz

### **Installation, for `macOS`** do the following: 


- Install the virtual environment and the required packages:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **Installation, for `WindowsOS`** do the following:

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