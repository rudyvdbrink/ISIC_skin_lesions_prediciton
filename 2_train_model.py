# %% get libraries
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn import metrics
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.applications import Xception

from supporting_functions import plot_images_grid_nometa

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import visualkeras

# %% definitions

model_name = 'Xception_multi-class_classifier' #how do we save the model when it's done
n_epochs_train    = 10
n_epochs_finetune = 4
#image_shape       = (71, 71) #full size is (450, 600)
#image_shape       = (225, 300) #full size is (450, 600)
image_shape       = (150, 200) #full size is (450, 600)


# %% load data

with open('data/processed/isic_data_allclasses.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

X_train            = loaded_data[0]
X_test             = loaded_data[1]
y_train            = loaded_data[2]
y_test             = loaded_data[3]
labels             = loaded_data[4]
metadata           = loaded_data[5]


# %%

# 5. Compute class weights to handle the class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))


# %% smart re-sample images

X_train = smart_resize(X_train,image_shape)
X_test  = smart_resize(X_test, image_shape)

plot_images_grid_nometa(X_train)

# %%

#one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train_encoded = to_categorical(y_train, num_classes)

# %% image augmentation (we over-sampled minority class)

# Create an ImageDataGenerator with random rotations and other augmentations
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

# %% freeze

base_model.trainable = False

# %% 

inputs = keras.Input(shape=(image_shape[0], image_shape[1], 3))

x = base_model(inputs, training=False) #leave the base model as is
x = keras.layers.GlobalAveragePooling2D()(x) #convert features of shape `base_model.output_shape[1:]` to vectors

# A Dense classifier with as many units as there are classes (multi-class classification)
outputs = keras.layers.Dense(num_classes)(x)
model = keras.Model(inputs, outputs)

# %% train the model

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
          epochs=n_epochs_train,
          class_weight=class_weights_dict)

# %% model fine tuning

# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

# Train end-to-end. Be careful to stop before you overfit!
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
          epochs=n_epochs_finetune,
           class_weight=class_weights_dict)


# %% make predictions

y_pred_proba = model.predict(X_test)
#y_pred_proba = np.concatenate((y_pred_proba*-1, y_pred_proba), axis=1)
y_pred       = np.argmax(y_pred_proba, axis=1)

# %% classification report and plots

print(classification_report(y_pred,y_test,zero_division=0))

print('Accuracy = ' + str(np.mean(y_pred==y_test)))
print('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,y_pred)))

# 3. Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred,normalize='true')
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, cmap="inferno", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,y_pred)))
plt.show()

# 4. Plot the ROC curves for each class
plt.figure(figsize=(5, 4))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(to_categorical(y_test, num_classes)[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {label} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# %% save model so that we don't have to run it again

model.save('./models/' + model_name + '.keras', overwrite=False)