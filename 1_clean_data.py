# %% import libraries

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

# %% load data

with open('data/imported/isic_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

data     = loaded_data[0]
metadata = loaded_data[1]

# %% drop information we cannot use, and split off the target

columns_to_keep = ['age_approx', 'anatom_site_general',  'sex', 'diagnosis']
metadata = metadata[columns_to_keep]
target = metadata.pop('diagnosis')

# %% label encode the target

labels = dict(enumerate(pd.factorize(target)[1]))
target = pd.factorize(target)[0]

# %% flatten data matrix

# Reshape the data from (11720, 450, 600, 3) to (11720, 810000)
data = data.reshape(data.shape[0], -1)

# # 2. Create a DataFrame from the flattened data
# flattened_df = pd.DataFrame(flattened_data)

# # 3. Concatenate the metadata with the flattened data horizontally
# # The resulting DataFrame will have metadata columns followed by flattened image data columns
# result_df = pd.concat([metadata, flattened_df], axis=1)

# %% max-scale data to unit height

print('Scaling data...')
for idx, im in enumerate(data):
    im = im / 255 #scale image to a maximum of 1
    data[idx] = im
print('Done!')

# chunk_size = 1000  # how many chunks do we want
# for i in range(0, data.shape[0], chunk_size): # Process the data in chunks
#     data[i:i+chunk_size] /= 255.0

# %% train test split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=1234)

# %% save data

data_to_save = [X_train, X_test, y_train, y_test, labels, metadata, data, target]

print('Saving data...')
# save data to file so we don't have to run it again
with open('data/processed/isic_data.pkl','wb') as f:
    pickle.dump(data_to_save,f) 
print('All done!')