# %% libraries

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from supporting_functions import evaluation_plots, load_dataset

# %% notes

# this code implements fitting of a base model, with a few things to note:
# - image re-scaling is applied to speed things up
# - image augmentation is applied so we over-fit less
# - class weights are included to make sure the model also learns less frequent classes

# %% user definitions

model_name        = 'Xception_multi-class_classifier_2019_0' #how do we save the model when it's done

data_dir          = './data/processed/2019_challenge/' # where did we store the images
target_shape      = (150, 200) #image shape after re-sizing

n_epochs_train    = 50
n_epochs_finetune = 10

# %% get .jpg data

train_ds, val_ds = load_dataset(data_dir)

# %% get the names of the classes
class_names = train_ds.class_names
num_classes = len(class_names)

# %% set up class weighting to deal with data imbalance

y_train = np.concatenate([y for _, y in train_ds], axis=0)
y_train = np.argmax(y_train,axis=1)
y_val= np.concatenate([y for _, y in val_ds], axis=0)
y_val= np.argmax(y_val,axis=1)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# %% define preprocessing layers

preprocessing_layers = tf.keras.Sequential([
  layers.Resizing(target_shape[0], target_shape[1]), #this re-scales the images to speed things up a bit
  layers.Rescaling(1./255), #this normalizes the color range to max = 1
  layers.RandomFlip("horizontal_and_vertical"), #randomly flip the image
  layers.RandomRotation(0.2), #randomly rotate
  layers.RandomTranslation(0.2, 0.2), #randomly shift horizontally and vertically
  layers.RandomZoom(0.2), #randomly zoom  
])

# %% define our (pre-trained) base model

base_model = Xception(
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
          class_weight=class_weights_dict,
          validation_data=val_ds
          )

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
          class_weight=class_weights_dict,
          validation_data=val_ds
          )

# %% evaluate the model

print('Training evaluation:')
evaluation_plots(model,train_ds)

print('Validation evaluation:')
evaluation_plots(model,val_ds)

# %% save model

model.save('./models/' + model_name + '.keras', overwrite=True)

# %% evaluate model on independent data

new_data_dir          = './data/processed/HAM10000/' # where did we store the images

test_ds = load_dataset(new_data_dir, full_set=1)

evaluation_plots(model, test_ds)
