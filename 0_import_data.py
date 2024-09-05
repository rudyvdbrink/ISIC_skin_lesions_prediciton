# %% libraries
import os
import shutil
import pandas as pd

print('Now copying over data...')

# %% definitions

#raw_dir       = './data/raw/HAM10000/'
#processed_dir = './data/processed/HAM10000/'

raw_dir       = './data/raw/2019_challenge/'
processed_dir = './data/processed/2019_challenge/'

# %% get metadata 

metadata = pd.read_csv(raw_dir + '/metadata.csv')

# %% re-organize images

#create target directory if it doesn't exist yet
os.makedirs(processed_dir, exist_ok=True)

#iterate through the dataframe and organize the images
for index, row in metadata.iterrows():
    label_dir = os.path.join(processed_dir, row.diagnosis) #target directory for the current file
    filename = row.isic_id + '.jpg' #name of the current file

    #create a subdirectory for the label if it doesn't exist yet
    os.makedirs(label_dir, exist_ok=True)   
    
    #define full source and target file paths
    src_file = os.path.join(raw_dir,   filename)
    dst_file = os.path.join(label_dir, filename)
    
    #copy the file
    shutil.copy(src_file, dst_file)

print('All done!')
