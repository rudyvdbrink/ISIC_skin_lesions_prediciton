# %% libraries

import tensorflow as tf
import keras

import numpy as np

from supporting_functions import evaluation_plots, retrieve_labels
from loading_functions import make_full_dataset_from_image_directory, make_balanced_split_dataset_from_image_directory

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# %% load model

#model_name = 'Xception_multi-class_classifier_pretrained' #what model to evaluate
model_name = 'Xception_multi-class_classifier_fully_trained_aggregate' #what model to evaluate
model = keras.saving.load_model("./models/" + model_name + ".keras")

# %% export as .tflite

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