# %% libraries
import os
from PIL import Image
import numpy as np
from supporting_functions import plot_images_grid

# %% read images
def load_images(folder_path):
    # List to store the images as numpy arrays
    images = []
    file_list = []

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # Check if the file is a .jpg image
            # Load the image
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Add the image array to the list
            images.append(img_array)
            file_list.append(filename)

    # Convert list of images to a single 3D numpy array
    images = np.array(images)

    return images, file_list

# load the images
folder_path  = './data/raw/'
data, file_names = load_images(folder_path)

print(data.shape)  # Prints the shape of the numpy array

# %% load meta data




# %% scale to max = 1

#data = data / 255

# %% plot some images to make sure it went correctly

plot_images_grid(data,start_N=0)
#plot_images_grid(data,start_N=0,num_images=4,grid_shape=(2,2))
