# %% get libraries
import numpy as np
import pickle
import time

from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16

import visualkeras

# %% load data

with open('data/processed/isic_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

X_train            = loaded_data[0]
X_test             = loaded_data[1]
y_train            = loaded_data[2]
y_test             = loaded_data[3]
labels             = loaded_data[4]
metadata           = loaded_data[5]

#%% reshape input data

#reshape X_train to the appropriate 4D shape (9376, 450, 600, 3)
# X_train = X_train.reshape((9376, 450, 600, 3))
# X_test  = X_test.reshape((2344, 450, 600, 3))

# %% sub-sample

X_train = X_train[:,::3,::3,:]
X_test  = X_test[:,::3,::3,:]


# %%

#one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train_encoded = to_categorical(y_train, num_classes)


# %% set up model

model = tf.keras.Sequential([
    tf.keras.Input(shape=(np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3])),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(), #turn output of previous layer into a vector rather than tensor
    tf.keras.layers.Dense(units=20,activation='relu'),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])

# %% info on model


model.summary()

#font = ImageFont.truetype("arial.ttf", 32) 
visualkeras.layered_view(model,legend=True, scale_xy=1, scale_z=0.01)


# %% image augmentation (we over-sampled )

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# %% compile and fit

#model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='Adam',loss='categorical_crossentropy')

# Train the model using the generator
model.fit(datagen.flow(X_train, to_categorical(y_train, num_classes=num_classes), batch_size=32),
          steps_per_epoch=len(X_train) // 32,
          epochs=20,
          )
model.summary()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
print('Accuracy = ' + str(np.mean(pred==y_test)))
print('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,pred)))

#plt.imshow(metrics.confusion_matrix(y_test,pred))
sns.heatmap(metrics.confusion_matrix(y_test,pred) / len(pred),linecolor='white',linewidths=0.05)
plt.xlabel("true label")
plt.ylabel("predicted label")
plt.title('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,pred)))
plt.show()

