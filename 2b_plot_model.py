
# %% get libraries
import numpy as np
import pickle
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16

from PIL import ImageFont

import visualkeras

# %% make model structure

num_classes = 8

# Load the VGG16 model with pre-trained ImageNet weights, without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(450, 600, 3))

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


# %% plot the model

font = ImageFont.truetype("arial.ttf", 32) 
visualkeras.layered_view(model,legend=True, legend_text_spacing_offset=20, font=font, scale_xy=1, scale_z=0.1).show() # display using your system viewer
visualkeras.layered_view(model,legend=True, legend_text_spacing_offset=20, font=font, scale_xy=1, scale_z=0.1, to_file='./figures/CNN.png') # write to disk
#visualkeras.layered_view(model, to_file='./figures/CNN.png').show() # write and show

#visualkeras.layered_view(model)

