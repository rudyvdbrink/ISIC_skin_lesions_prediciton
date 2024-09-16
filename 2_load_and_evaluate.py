# %% libraries

import tensorflow as tf
import keras

import numpy as np

from supporting_functions import evaluation_plots, retrieve_labels
from loading_functions import make_full_dataset_from_image_directory, make_balanced_split_dataset_from_image_directory

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# %% definitions

#model_name = 'Xception_multi-class_classifier_pretrained' #what model to evaluate
model_name = 'Xception_multi-class_classifier_fully_trained_aggregate' #what model to evaluate
#model_name = 'InceptionResNetV2_multi-class_classifier_fully_trained' #what model to evaluate

data_dir   = './data/processed/HAM10000/' # where did we store the images
#data_dir   = './data/processed/2019_challenge/' # where did we store the images


# %% load data

#ds = make_full_dataset_from_image_directory(data_dir,batch_size=32,shuffle=True)

batch_size     = 32
validation     = True
shuffle        = True
split          = [0.7, 0.1, 0.2] #train, validation, test split proportions

train_ds, _, test_ds = make_balanced_split_dataset_from_image_directory(data_dir, 
                                                                 batch_size, 
                                                                 None, 
                                                                 split=split, 
                                                                 validation=validation, 
                                                                 shuffle=shuffle)

# %% load model

model = keras.saving.load_model("./models/" + model_name + ".keras")

# %%

evaluation_plots(model, train_ds)

# %% make evaluation plots

evaluation_plots(model, test_ds)

# %% examine accuracy on the binary problem

# y = retrieve_labels(test_ds)
# y = np.where(np.isin(y, [0, 1, 3, 6]), 1, y)
# y = np.where(np.isin(y, [2, 4, 5, 7]), 0, y)

# pred = model.predict(test_ds)
# y_pred = np.argmax(pred,axis=1)
# y_pred = np.where(np.isin(y_pred, [0, 1, 3, 6]), 1, y_pred)
# y_pred = np.where(np.isin(y_pred, [2, 4, 5, 7]), 0, y_pred)

# print('Binary accraucy = ' + str(sum(y_pred == y)/len(y)))
# print('Binary balanced accraucy = ' + str(balanced_accuracy_score(y,y_pred)))

# cfm = confusion_matrix(y,y_pred,normalize='true')
# sns.heatmap(cfm,annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('True')

# %% examine accuracy without the 8th class

# y      = retrieve_labels(test_ds)
# y_pred = np.argmax(pred,axis=1)

# idx = np.where(y == 0)[0]

# y      = np.delete(y, idx)
# y_pred = np.delete(y_pred, idx)

# print('Class-corrected accraucy = ' + str(sum(y_pred == y)/len(y)))
# print('Class-corrected balanced accraucy = ' + str(balanced_accuracy_score(y,y_pred)))