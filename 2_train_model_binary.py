# %% get libraries
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import smart_resize

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

# X_train = X_train[:,::3,::3,:]
# X_test  = X_test[:,::3,::3,:]

# %% smart re-sample images

# X_train = smart_resize(X_train,(28, 28))
# X_test  = smart_resize(X_test,(28, 28))

X_train = smart_resize(X_train,(56, 56))
X_test  = smart_resize(X_test,(56, 56))

#plot_images_grid_nometa(X_train)

# %%

#one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train_encoded = to_categorical(y_train, num_classes)


# %% set up model

model = tf.keras.Sequential([
    tf.keras.Input(shape=(np.shape(X_train)[1], np.shape(X_train)[2], np.shape(X_train)[3])),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
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


# pred = model.predict(X_test)
# pred = np.argmax(pred,axis=1)
# print('Accuracy = ' + str(np.mean(pred==y_test)))
# print('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,pred)))

# #plt.imshow(metrics.confusion_matrix(y_test,pred))
# sns.heatmap(metrics.confusion_matrix(y_test,pred) / len(pred),linecolor='white',linewidths=0.05)
# plt.xlabel("true label")
# plt.ylabel("predicted label")
# plt.title('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,pred)))
# plt.show()


# %%

# 1. Evaluate the model on the test data
num_classes = len(np.unique(y_test))
# test_loss, test_accuracy = model.evaluate(X_test, to_categorical(y_test, num_classes), verbose=0)
# print(f'Test Loss: {test_loss:.4f}')
# print(f'Test Accuracy: {test_accuracy:.4f}')

# 2. Predict the class probabilities and class labels
y_pred_proba = model.predict(X_test)
y_pred       = np.argmax(y_pred_proba, axis=1)

# %% classification report

print(classification_report(y_pred,y_test,zero_division=0))

# %% plots

print('Accuracy = ' + str(np.mean(y_pred==y_test)))
print('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,y_pred)))

# 3. Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred,normalize='true')
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, cmap="inferno", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y_test,pred)))
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
