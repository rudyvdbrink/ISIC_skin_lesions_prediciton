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
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.applications import VGG16

#from supporting_functions import plot_images_grid_nometa
from supporting_functions import evaluation_plots, retrieve_data, retrieve_labels
from loading_functions import make_balanced_dataset_from_image_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% definitions

model_name = 'VGG16_multi-class_classifier_pretrained_aggregate' #how do we save the model when it's done
n_epochs_train    = 3
n_epochs_finetune = 2
image_shape       = (150, 200) #full size is (450, 600)

# %% load data

data_dir          = './data/processed/Aggregate/' # where did we store the images
target_shape      = (150, 200) #image shape after re-sizing
batch_size        = 32
target_size       = [6200, 6200, 6200, 6200, None, 6200, 6200, 6200]
#target_size       = 100


train_ds = make_balanced_dataset_from_image_directory(data_dir, 
                                                 batch_size=batch_size, 
                                                 target_size=target_size,
                                                 shuffle=False)


# %% set up class weighting to deal with data imbalance
class_names = train_ds.class_names
num_classes = len(class_names)

y_train = np.concatenate([y for _, y in train_ds], axis=0)
y_train = np.argmax(y_train,axis=1)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# %% define preprocessing layers

preprocessing_layers = tf.keras.Sequential([
  layers.Resizing(target_shape[0], target_shape[1]), #this re-scales the images to speed things up a bit
  layers.RandomFlip("horizontal_and_vertical"), #randomly flip the image
  layers.RandomRotation(0.2), #randomly rotate
  layers.RandomTranslation(0.2, 0.2), #randomly shift horizontally and vertically
  layers.RandomZoom(0.2), #randomly zoom  
])

# %% define our (pre-trained) base model

base_model = VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(target_shape[0], target_shape[1], 3),
    include_top=False)  # Do not include the ImageNet classifier at the top

base_model.trainable = False #freeze the base model

# %%  build our model

#pass the inputs through the preprocessing layers 
inputs  = keras.Input(shape=(None, None, 3))  #input layer that matches the original image size
x       = preprocessing_layers(inputs) #add our custom pre-processing layers
x       = base_model(x)  # add base model
x       = keras.layers.GlobalAveragePooling2D()(x) #global average pooling to convert features from base model to vectors
outputs = keras.layers.Dense(num_classes)(x) #add a Dense layer for classification
model   = keras.Model(inputs, outputs) #create the complete model

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

# %% run the initial model fit

model.fit(train_ds,
          epochs=n_epochs_train, 
          class_weight=class_weights_dict)

# %% model fine tuning

#unfreeze the base model
base_model.trainable = True

#recompile
model.compile(optimizer=keras.optimizers.Adam(1e-5),  #very low learning rate
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

#run fine-tuning fit
model.fit(train_ds,
          epochs=n_epochs_finetune,
          class_weight=class_weights_dict)

# %% save model 

#model.save('./models/' + model_name + '.keras', overwrite=False)

# %% evalulate (training data)

evaluation_plots(model, train_ds)