# %% import libraries

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from supporting_functions import plot_images_grid_nometa

from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample

from tensorflow.keras.datasets import mnist

# %% load mnist data

(X_train, y_train), (X_test, y_test) = mnist.load_data() 


# %% rebalance data (super-sample less frequent class)

# Flatten the input data temporarily to use with RandomOverSampler
X_train_flat = X_train.reshape((X_train.shape[0], -1))  # Reshape to (9376, 90000)


# Apply RandomOverSampler to balance the classes
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)

# Reshape X_train back to the original shape
X_train_resampled = X_train_resampled.reshape((-1, X_train.shape[1], X_train.shape[2]))

# Verify the class distribution after resampling
unique, counts = np.unique(y_train_resampled, return_counts=True)
print(f'Class distribution after resampling: {dict(zip(unique, counts))}')

# %% sub-sample data to its original size

# Sub-sample the resampled data to match the original size
X_train_subsampled, y_train_subsampled = resample(
    X_train_resampled, y_train_resampled, 
    replace=False,  # Do not replace, we want exactly original size
    n_samples=X_train.shape[0],  # Original size
    random_state=42  # For reproducibility
)

# Verify the shape
print(f"Sub-sampled X_train shape: {X_train_subsampled.shape}")
print(f"Sub-sampled y_train shape: {y_train_subsampled.shape}")


# %% over-write original variables and save

X_train = X_train_subsampled
y_train = y_train_subsampled

# %% save data

data_to_save = [X_train, X_test, y_train, y_test]

print('Saving data...')
# save data to file so we don't have to run it again
with open('../data/processed/mnist_data.pkl','wb') as f:
    pickle.dump(data_to_save,f) 
print('All done!')

#plot_images_grid(data,metadata_full,start_N=0)

#%%

plot_images_grid_nometa(X_train,start_N=0)

