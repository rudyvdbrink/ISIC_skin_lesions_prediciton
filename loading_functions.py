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
def make_dataset_from_image_sub_directory(sub_dir, label):

    files = glob.glob(sub_dir + '/*.jpg') #list of image file paths
    
    labels = np.zeros( (len(files), 8)) #one-hot encoded labels
    labels[:,label] = 1

    #convert the file paths and labels to TensorFlow datasets
    sub_dir_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds  = tf.data.Dataset.from_tensor_slices(labels)

    #apply the function to load images from file paths
    image_ds = sub_dir_ds.map(load_and_preprocess_image)

    #combine images and labels into a single dataset 
    ds = tf.data.Dataset.zip((image_ds, labels_ds))

    return ds

#function for making a balanced dataset out of one directory, with a specific size
def make_balanced_dataset_from_image_sub_directory(sub_dir, label, batch_size, target_size=None):

    files = glob.glob(sub_dir + '/*.jpg') #list of image file paths

    if target_size is None:
        target_size = len(files)

    labels = np.zeros((target_size, 8)) #one-hot encoded labels
    labels[:, label] = 1

    # balance the dataset
    data_size = len(files)
    N = target_size - data_size

    if N > 0:  # if we want to over-sample
        # randomly select N files from the list and add
        additional_files = random.choices(files, k=N)
        files.extend(additional_files)
    else:  # if we want to under-sample
        files = random.sample(files, target_size)

    # convert the file paths and labels to TensorFlow datasets
    sub_dir_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    # apply the function to load images from file paths
    image_ds = sub_dir_ds.map(load_and_preprocess_image)

    # combine images and labels into a single dataset
    ds = tf.data.Dataset.zip((image_ds, labels_ds))

    return ds

#function for loading full dataset
def make_full_dataset_from_image_directory(base_dir, batch_size, shuffle=True):

    #get list of all subdirectories (one for each class)
    sub_dirs = [x[0] for x in os.walk(base_dir) if '/' not in x[0][-1]]
    class_names = [x[0].split('/')[-1] for x in os.walk(base_dir) if '/' not in x[0][-1]]

    for label, sub_dir in enumerate(sub_dirs):  
        if label == 0:      
            ds = make_dataset_from_image_sub_directory(sub_dir, label)
        else: 
            subds = make_dataset_from_image_sub_directory(sub_dir, label)
            ds = ds.concatenate(subds)

    ds.class_names = class_names
    #shuffle, batch, and prefetch the dataset as needed
    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality().numpy()*batch_size, seed=42, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds.class_names = class_names

    return ds

#function to make lists for train-test split data
def split_files(files, split, validation=True):
    random.seed(42)
    random.shuffle(files)
    if validation:
        #calculate split indices
        train_idx = int(len(files) * split[0])
        val_idx = train_idx + int(len(files) * split[1])

        #split the files list into train, val, and test
        train_files = files[:train_idx]
        val_files   = files[train_idx:val_idx]
        test_files  = files[val_idx:]

        return train_files, val_files, test_files
    else:
        #calculate split indices
        train_idx = int(len(files) * split[0])
        #split the files list into train and test
        train_files = files[:train_idx]
        test_files  = files[train_idx:]

        return train_files, test_files

#function for building a dataset given a list of images and labels
def build_dataset(files,labels):
    #convert the file paths and labels to TensorFlow datasets
    sub_dir_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds  = tf.data.Dataset.from_tensor_slices(labels)

    #apply the function to load images from file paths
    image_ds = sub_dir_ds.map(load_and_preprocess_image)

    #combine images and labels into a single dataset 
    return tf.data.Dataset.zip((image_ds, labels_ds))

#function for building a balanced dataset with train-test split 
def make_balanced_split_dataset_from_image_sub_directory(sub_dir, label,                                                          
                                                         target_size=None, 
                                                         split=[0.7, 0.1, 0.2], 
                                                         validation=True ):
    
    files = glob.glob(sub_dir + '/*.jpg') #list of image file paths

    if target_size is None:
        target_size = len(files)

    #make a train-test split for the file lists
    if validation:
        train_files, val_files, test_files = split_files(files, split, validation)
    else:
        train_files, test_files = split_files(files, split, validation)
    
    train_labels = np.zeros((target_size, 8)) #one-hot encoded labels
    train_labels[:, label] = 1

    if validation:
        val_labels = np.zeros((len(val_files), 8)) #one-hot encoded labels
        val_labels[:, label] = 1

    test_labels = np.zeros((len(test_files), 8)) #one-hot encoded labels
    test_labels[:, label] = 1
    
    #balance the training dataset
    data_size = len(train_files)
    N = target_size - data_size

    if N > 0: #if we want to over-sample
        additional_files = random.choices(train_files, k=N)
        train_files.extend(additional_files)
    else: #if we want to under-sample
        train_files = random.sample(train_files, target_size)

    #now build the datasets
    train_ds = build_dataset(train_files, train_labels)

    if validation:
        val_ds = build_dataset(val_files, val_labels)

    test_ds = build_dataset(test_files, test_labels)
    
    if validation:
        return train_ds, val_ds, test_ds
    else:
        return train_ds, test_ds

#function for loading a balanced dataset from directory
def make_balanced_dataset_from_image_directory(base_dir, batch_size, target_size=None, shuffle=True):

    sub_dirs = [x[0] for x in os.walk(base_dir) if '/' not in x[0][-1]]
    class_names = [x[0].split('/')[-1] for x in os.walk(base_dir) if '/' not in x[0][-1]]

    for label, sub_dir in enumerate(sub_dirs):  
        if isinstance(target_size, list):
            current_target_size = target_size[label]
        else:
            current_target_size = target_size

        if label == 0:      
            ds = make_balanced_dataset_from_image_sub_directory(sub_dir, label, batch_size, current_target_size)
        else: 
            subds = make_balanced_dataset_from_image_sub_directory(sub_dir, label, batch_size, current_target_size)
            ds = ds.concatenate(subds)

    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality().numpy() * batch_size, seed=42, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds.class_names = class_names

    return ds

#function for loading a balanced dataset with train-test-split
def make_balanced_split_dataset_from_image_directory(base_dir, batch_size, target_size=None, split=[0.7, 0.1, 0.2], validation=True, shuffle=True):

    sub_dirs = [x[0] for x in os.walk(base_dir) if '/' not in x[0][-1]]
    class_names = [x[0].split('/')[-1] for x in os.walk(base_dir) if '/' not in x[0][-1]]

    for label, sub_dir in enumerate(sub_dirs):

        if isinstance(target_size, list):
            current_target_size = target_size[label]
        else:
            current_target_size = target_size

        if validation:
            if label == 0:
                train_ds, val_ds, test_ds = make_balanced_split_dataset_from_image_sub_directory(sub_dir, label, current_target_size, split, validation)
            else:
                sub_train_ds, sub_val_ds, sub_test_ds = make_balanced_split_dataset_from_image_sub_directory(sub_dir, label, current_target_size, split, validation)
                train_ds = train_ds.concatenate(sub_train_ds)
                val_ds = val_ds.concatenate(sub_val_ds)
                test_ds = test_ds.concatenate(sub_test_ds)
        else:
            if label == 0:
                train_ds, test_ds = make_balanced_split_dataset_from_image_sub_directory(sub_dir, label, current_target_size, split, validation)
            else:
                sub_train_ds, sub_test_ds = make_balanced_split_dataset_from_image_sub_directory(sub_dir, label, current_target_size, split, validation)
                train_ds = train_ds.concatenate(sub_train_ds)
                test_ds = test_ds.concatenate(sub_test_ds)

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=train_ds.cardinality().numpy() * batch_size, seed=42, reshuffle_each_iteration=False)
    train_ds = train_ds.batch(batch_size=batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_ds.class_names = class_names

    if validation:
        if shuffle:
            val_ds = val_ds.shuffle(buffer_size=val_ds.cardinality().numpy() * batch_size, seed=42, reshuffle_each_iteration=False)
        val_ds = val_ds.batch(batch_size=batch_size)
        val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_ds.class_names = class_names

    if shuffle:
        test_ds = test_ds.shuffle(buffer_size=test_ds.cardinality().numpy() * batch_size, seed=42, reshuffle_each_iteration=False)
    test_ds = test_ds.batch(batch_size=batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds.class_names = class_names

    if validation:
        return train_ds, val_ds, test_ds
    else:
        return train_ds, test_ds
        