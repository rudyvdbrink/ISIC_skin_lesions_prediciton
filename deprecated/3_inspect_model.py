# %% libraries

import tensorflow as tf
import keras

import numpy as np

from supporting_functions import evaluation_plots, retrieve_labels, retrieve_data
from loading_functions import make_full_dataset_from_image_directory, make_balanced_split_dataset_from_image_directory

import shap
import random

# %% definitions

#model_name = 'Xception_multi-class_classifier_pretrained' #what model to evaluate
model_name = 'Xception_multi-class_classifier_fully_trained' #what model to evaluate

data_dir   = './data/processed/HAM10000/' # where did we store the images
#data_dir   = './data/processed/2019_challenge/' # where did we store the images


# %% load data

#ds = make_full_dataset_from_image_directory(data_dir,batch_size=32,shuffle=True)

batch_size     = 32
validation     = True
shuffle        = True
split          = [0.7, 0.1, 0.2] #train, validation, test split proportions

_, _, test_ds = make_balanced_split_dataset_from_image_directory(data_dir, 
                                                                 batch_size, 
                                                                 10, 
                                                                 split=split, 
                                                                 validation=validation, 
                                                                 shuffle=shuffle)

# %% load model

model = keras.saving.load_model("./models/" + model_name + ".keras")
# pred = model.predict(test_ds)

# %% 

y = retrieve_labels(test_ds)
X = retrieve_data(test_ds)

# %%

random_indices = np.random.choice(X.shape[0], 100, replace=False)
images_to_explain = X[random_indices]

from supporting_functions import plot_images_grid_nometa
plot_images_grid_nometa(images_to_explain)

# %% 

explainer = shap.DeepExplainer(model, X[:100])

#explainer = shap.KernelExplainer(model.predict, X[:100])


# Compute SHAP values for the selected subset
shap_values = explainer.shap_values(images_to_explain)
# Visualize the first prediction's explanation
shap.image_plot(shap_values, -images_to_explain[:1])