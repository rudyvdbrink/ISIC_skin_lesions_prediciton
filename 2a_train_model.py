# %% get libraries
import numpy as np
import pickle
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16

# %% load data

with open('data/processed/isic_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

X_train            = loaded_data[0]
X_test             = loaded_data[1]
y_train            = loaded_data[2]
y_test             = loaded_data[3]
labels             = loaded_data[4]
metadata           = loaded_data[5]

# %% set up model

# 1. Reshape X_train to the appropriate 4D shape (9376, 450, 600, 3)
X_train = X_train.reshape((9376, 450, 600, 3))

# 2. One-hot encode the labels
num_classes = len(np.unique(y_train))
y_train_encoded = to_categorical(y_train, num_classes)

# Load the VGG16 model with pre-trained ImageNet weights, without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(450, 600, 3))

# Freeze the base model layers to prevent them from being updated during training
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

# Combine base model with custom layers
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Compute class weights to handle the class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# %% fit

# Start the timer
start_time = time.time()

# Fit the model
model.fit(X_train, y_train_encoded, 
          epochs=20, 
          batch_size=32, 
          class_weight=class_weights_dict,
          validation_split=0.2)

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print('Fitting took ' + str(elapsed_time / 60) + ' minutes')

# Print the model summary
model.summary()

# %% save the model

# save model to file so we don't have to run it again
with open('models/CNN_classifier.pkl','wb') as f:
    pickle.dump(model,f)    

# %%

# # 3. Build custom CNN model
# model = Sequential()

# # Convolutional layers
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(450, 600, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Flatten the output of the convolutional layers
# model.add(Flatten())

# # Fully connected layers
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))

# # Output layer
# model.add(Dense(num_classes, activation='softmax'))

# # 4. Compile the model
# model.compile(optimizer='adam', 
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])