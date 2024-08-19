# %% libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
from supporting_functions import plot_images_grid
import pickle

# %% read images
def load_images(raw_images_dir):
    # List to store the images as numpy arrays
    images = []
    file_names = []

    print('Now loading data...')

    # Iterate over the files in the folder
    for filename in os.listdir(raw_images_dir):
        if filename.endswith('.jpg'):  # Check if the file is a .jpg image
            # Load the image
            img_path = os.path.join(raw_images_dir, filename)
            img = Image.open(img_path)
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Add the image array to the list
            images.append(img_array)
            file_names.append(filename)

    # Convert list of images to a single 3D numpy array
    images = np.array(images)
    print('Data loaded successfully')

    return images, file_names

# %% function to load metadata


def load_and_sort_metadata(csv_path, file_names):
    # Load the metadata from the .csv file
    metadata = pd.read_csv(csv_path + 'ham10000_metadata_2024-08-17.csv')

    # Get the list of image file names (without the .jpg suffix)
    image_filenames = [filename.split('.')[0] for filename in file_names if filename.endswith('.jpg')]

    # Ensure the metadata is sorted according to the image filenames order
    metadata_sorted = metadata.set_index('isic_id').loc[image_filenames].reset_index()

    return metadata_sorted

if __name__ == '__main__':
    # %% folder defintions

    raw_images_dir  = './data/raw/'
    metadata_dir    = './data/metadata/'


    # %% perform tasks

    # load the images
    data, file_names = load_images(raw_images_dir)

    #print(data.shape)  # Prints the shape of the numpy array

    # %% load meta data

    # Example usage
    metadata = load_and_sort_metadata(metadata_dir, file_names)

    # Display the first few rows of the sorted metadata
    #print(metadata.head())


    # %% scale to max = 1

    #data = data / 255

    # %% plot some images to make sure it went correctly

    #plot_images_grid(data,metadata,start_N=0)
    #plot_images_grid(data,start_N=0,num_images=4,grid_shape=(2,2))

    # %% save data to file

    data_to_save = [data, metadata, file_names]

    # save data to file so we don't have to run it again
    with open('data/imported/isic_data.pkl','wb') as f:
        pickle.dump(data_to_save,f) 


    print('All done!')
