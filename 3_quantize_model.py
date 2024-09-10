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
model_name = 'Xception_multi-class_classifier_fully_trained_3rounds' #what model to evaluate

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
                                                                 1000, 
                                                                 split=split, 
                                                                 validation=validation, 
                                                                 shuffle=shuffle)

# %% load model

model = keras.saving.load_model("./models/" + model_name + ".keras")

# %%

# Convert the model to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations for quantization (e.g., weight quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optionally, if you have a representative dataset for full integer quantization:
# def representative_data_gen():
#     for input_value in dataset:  # Replace 'dataset' with your data
#         yield [input_value]

# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Convert the model
tflite_model = converter.convert()

# Save the quantized model
tflite_model_path = f"./models/{model_name}_quantized.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Quantized model saved to {tflite_model_path}")