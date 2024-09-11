# %% get libraries
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.applications import Xception

#from supporting_functions import plot_images_grid_nometa
from supporting_functions import evaluation_plots, retrieve_data, retrieve_labels
from loading_functions import make_balanced_dataset_from_image_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% definitions

model_name = 'Xception_multi-class_classifier_pretrained' #how do we save the model when it's done
n_epochs_train    = 10
n_epochs_finetune = 5
image_shape       = (150, 200) #full size is (450, 600)

# %% load data

data_dir          = './data/processed/2019_challenge/' # where did we store the images
target_shape      = (150, 200) #image shape after re-sizing
n_epochs_train    = 10
batch_size        = 32
target_size       = 6200

train_ds = make_balanced_dataset_from_image_directory(data_dir, 
                                                 batch_size=batch_size, 
                                                 target_size=target_size,
                                                 shuffle=False)

X_train = retrieve_data(train_ds)
y_train = retrieve_labels(train_ds)

# %% compute class weights to handle the class imbalance

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# %% #one-hot encode the labels

num_classes = len(np.unique(y_train))
y_train_encoded = to_categorical(y_train, num_classes)

# %% image augmentation (we over-sampled minority class)

#create an ImageDataGenerator with random rotations and other augmentations
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
#fit the generator on data
datagen.fit(X_train)

#  %% load a base model

base_model = Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(image_shape[0], image_shape[1], 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

# %% freeze the base model

base_model.trainable = False

# %% add in custom input and output

inputs = keras.Input(shape=(image_shape[0], image_shape[1], 3))

x = base_model(inputs, training=False) #leave the base model as is
x = keras.layers.GlobalAveragePooling2D()(x) #convert features of shape `base_model.output_shape[1:]` to vectors

#a Dense classifier with as many units as there are classes (multi-class classification)
outputs = keras.layers.Dense(num_classes)(x)
model = keras.Model(inputs, outputs)

# %% train the model (only the added layer)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
          epochs=n_epochs_train,
          class_weight=class_weights_dict)

# %% model fine tuning (end-to-end training)

#unfreeze the base model
base_model.trainable = True

#recompile your model so that changes are take into account
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # very low learning rate
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

#train end-to-end
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
          epochs=n_epochs_finetune,
           class_weight=class_weights_dict)

# %% save model so that we can run it again

model.save('./models/' + model_name + '.keras', overwrite=False)

# %% evalulate (training data)

evaluation_plots(model, train_ds)

# %% load testing data

data_dir          = './data/processed/HAM10000/' # where did we store the images
target_size       = 1000

test_ds = make_balanced_dataset_from_image_directory(data_dir, 
                                                 batch_size=batch_size, 
                                                 target_size=target_size,
                                                 shuffle=False)

# %% evalulate (testing data)

evaluation_plots(model, test_ds)