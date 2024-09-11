# %% libraries
import os
import shutil
import pandas as pd

print('Now copying over data...')

# %% definitions

#raw_dir       = './data/raw/HAM10000/'
#processed_dir = './data/processed/HAM10000/'

# raw_dir       = './data/raw/2019_challenge/'
# processed_dir = './data/processed/2019_challenge/'

raw_dir       = './data/raw/Diverse_derm/'
processed_dir = './data/processed/Diverse_derm/'


# %% get metadata 

metadata = pd.read_csv(raw_dir + '/metadata.csv')

# %% define disease mapping for consistency
# Create a mapping dictionary
disease_mapping = {
    'melanoma-in-situ': 'melanoma',
    'melanoma-acral-lentiginous': 'melanoma',
    'nodular-melanoma-(nm)': 'melanoma',
    'melanoma': 'melanoma',
    'atypical-spindle-cell-nevus-of-reed': 'nevus',
    'pigmented-spindle-cell-nevus-of-reed': 'nevus',
    'blue-nevus': 'nevus',
    'nevus-lipomatosus-superficialis': 'nevus',
    'melanocytic-nevi': 'nevus',
    'dysplastic-nevus': 'nevus',
    'actinic-keratosis': 'actinic keratosis',
    'basal-cell-carcinoma': 'basal cell carcinoma',
    'basal-cell-carcinoma-superficial': 'basal cell carcinoma',
    'basal-cell-carcinoma-nodular': 'basal cell carcinoma',
    'squamous-cell-carcinoma': 'squamous cell carcinoma',
    'squamous-cell-carcinoma-in-situ': 'squamous cell carcinoma',
    'squamous-cell-carcinoma-keratoacanthoma': 'squamous cell carcinoma',
    'vascular-lesion': 'vascular lesion',  # No clear match, assuming this is vascular
    'dermatofibroma': 'dermatofibroma',
    'seborrheic-keratosis': 'pigmented benign keratosis',
    'seborrheic-keratosis-irritated': 'pigmented benign keratosis',
    'solar-lentigo': 'pigmented benign keratosis',
    #the rest go to 'other'
}

# Function to map diseases to categories
def map_disease_to_category(disease):
    return disease_mapping.get(disease, 'other')

# Apply the mapping to the 'disease' column
metadata['diagnosis'] = metadata['disease'].apply(map_disease_to_category)

# %% re-organize images

#create target directory if it doesn't exist yet
os.makedirs(processed_dir, exist_ok=True)

#iterate through the dataframe and organize the images
for index, row in metadata.iterrows():
    label_dir = os.path.join(processed_dir, row.diagnosis) #target directory for the current file
    filename = row.DDI_file #name of the current file

    #create a subdirectory for the label if it doesn't exist yet
    os.makedirs(label_dir, exist_ok=True)   
    
    #define full source and target file paths
    src_file = os.path.join(raw_dir,   filename)
    dst_file = os.path.join(label_dir, filename)
    
    #copy the file
    shutil.copy(src_file, dst_file)

print('All done!')
