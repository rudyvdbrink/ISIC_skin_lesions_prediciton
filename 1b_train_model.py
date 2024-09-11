# %% libraries

import tensorflow as tf
import keras
import numpy as np

from supporting_functions import evaluation_plots, retrieve_data, retrieve_labels
from loading_functions import make_balanced_split_dataset_from_image_directory

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# %% definitions

load_model_name = 'Xception_multi-class_classifier_pretrained_aggregate' #what model to load
#save_model_name = 'Xception_multi-class_classifier_fully_trained_aggregate_2rounds' #how to save the model afterwards

base_dir       = './data/processed/HAM10000/'
batch_size     = 32
validation     = True
shuffle        = True
split          = [0.7, 0.1, 0.2] #train, validation, test split proportions
target_size    = [6200, 6200, 6200, 6200, None, 6200, 6200, 6200]
n_epochs_train = 2


# %% get data

train_ds, val_ds, test_ds = make_balanced_split_dataset_from_image_directory(base_dir, 
                                                                            batch_size, 
                                                                            target_size, 
                                                                            split=split, 
                                                                            validation=validation, 
                                                                            shuffle=shuffle)

# X_train = retrieve_data(train_ds)
# y_train = retrieve_labels(train_ds)

y_train = retrieve_labels(train_ds)

#create tensors for fitting image augmentation
X = tf.data.Dataset.unbatch(train_ds)
X_train, y = zip(*X)

# %% load model to continue training

model = keras.saving.load_model("./models/" + load_model_name + ".keras")

# %% make evaluation plots before training

print('Pre-training evaluation on test-set: ')
evaluation_plots(model, test_ds)

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
#datagen.fit(X)

# %% train end-to-end

# model.fit( datagen.flow(X_train, y_train_encoded, batch_size=batch_size),
#            epochs=n_epochs_train,
#            class_weight=class_weights_dict,
#            validation_data=val_ds
#            )
model.fit( datagen.flow(np.array(X_train), y, batch_size=batch_size),
           epochs=n_epochs_train,
           class_weight=class_weights_dict,
           validation_data=val_ds
           )

# %% save model so that we can run it again

model.save('./models/' + save_model_name + '.keras', overwrite=False)

# %% evalulate (testing data)

print('Post-training evaluation on test-set: ')
evaluation_plots(model, test_ds)