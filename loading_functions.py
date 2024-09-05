# %% libraries
import os
import glob
import tensorflow as tf
import numpy as np
import random

# %% custom data loading functions

#function to load and preprocess images
def load_and_preprocess_image(file_path):
    #load the image from file
    image = tf.io.read_file(file_path)
    #decode the image as JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    #resize or preprocess the image
    image = tf.image.resize(image, [150, 200])    
    #image = tf.image.resize(image, [450, 600])    
    #image = image / 255.0
    return image

#function for making a dataset out of one directory
def make_dataset_from_image_sub_directory(sub_dir, label, batch_size):

    files = glob.glob(sub_dir + '/*.jpg') #list of image file paths
    #labels = np.zeros(len(files)) + label #corresponding labels (same for each file)
    
    labels = np.zeros( (len(files), 8)) #one-hot encoded labels
    labels[:,label] = 1

    #convert the file paths and labels to TensorFlow datasets
    sub_dir_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds  = tf.data.Dataset.from_tensor_slices(labels)

    #apply the function to load images from file paths
    image_ds = sub_dir_ds.map(load_and_preprocess_image)

    #combine images and labels into a single dataset 
    ds = tf.data.Dataset.zip((image_ds, labels_ds))

    #shuffle, batch, and prefetch the dataset as needed
    #ds = ds.shuffle(buffer_size=len(files), seed=42)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

#function for making a dataset out of one directory, with a specific size
def make_balanced_dataset_from_image_sub_directory(sub_dir, label, batch_size, target_size):

    files = glob.glob(sub_dir + '/*.jpg') #list of image file paths
    #labels = np.zeros(len(files)) + label #corresponding labels (same for each file)

    labels = np.zeros( (target_size, 8)) #one-hot encoded labels
    labels[:,label] = 1

    #balance the dataset
    data_size = len(files)
    N = target_size - data_size

    if N > 0: #if we want to over-sample
        #fandomly select N files from the list and add
        additional_files = random.choices(files, k=N)
        files.extend(additional_files)
    else: #if we want to under-sample
        files = random.sample(files, target_size)

    #convert the file paths and labels to TensorFlow datasets
    sub_dir_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds  = tf.data.Dataset.from_tensor_slices(labels)

    #apply the function to load images from file paths
    image_ds = sub_dir_ds.map(load_and_preprocess_image)

    #combine images and labels into a single dataset 
    ds = tf.data.Dataset.zip((image_ds, labels_ds))    

    return ds

#function for loading full dataset
def make_full_dataset_from_image_directory(base_dir, batch_size, shuffle=True):

    #get list of all subdirectories (one for each class)
    sub_dirs = [x[0] for x in os.walk(base_dir) if '/' not in x[0][-1]]
    class_names = [x[0].split('/')[-1] for x in os.walk(base_dir) if '/' not in x[0][-1]]

    for label, sub_dir in enumerate(sub_dirs):  
        if label == 0:      
            ds = make_dataset_from_image_sub_directory(sub_dir, label, batch_size)
        else: 
            subds = make_dataset_from_image_sub_directory(sub_dir, label, batch_size)
            ds = ds.concatenate(subds)

    ds.class_names = class_names
    #shuffle, batch, and prefetch the dataset as needed
    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality().numpy()*batch_size, seed=42)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds.class_names = class_names

    return ds

#function for loading full dataset
def make_balanced_dataset_from_image_directory(base_dir, batch_size, target_size, shuffle=True):

    #get list of all subdirectories (one for each class)
    sub_dirs = [x[0] for x in os.walk(base_dir) if '/' not in x[0][-1]]
    class_names = [x[0].split('/')[-1] for x in os.walk(base_dir) if '/' not in x[0][-1]]

    for label, sub_dir in enumerate(sub_dirs):  
        if label == 0:      
            ds = make_balanced_dataset_from_image_sub_directory(sub_dir, label, batch_size, target_size)
        else: 
            subds = make_balanced_dataset_from_image_sub_directory(sub_dir, label, batch_size, target_size)
            ds = ds.concatenate(subds)

    #shuffle, batch, and prefetch the dataset as needed
    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality().numpy()*batch_size, seed=42)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds.class_names = class_names

    return ds

