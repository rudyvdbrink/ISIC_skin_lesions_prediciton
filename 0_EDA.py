# %% libraries

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# %% definitions

#set_name = 'HAM10000'
set_name = '2019_challenge'

raw_dir       = './data/raw/' + set_name + '/'
processed_dir = './data/processed/' + set_name + '/'

# %% get .jpg data

data = tf.keras.preprocessing.image_dataset_from_directory(
    processed_dir,
    labels='inferred',
    image_size=(450, 600),
    batch_size=32,
    shuffle=True
)

class_names = data.class_names

# %% plot some random images

plt.figure(figsize=(9, 7))
for images, labels in data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %% calculate the frequency per class

# Extract all labels from the dataset
all_labels = []
for images, labels in data:
    all_labels.extend(labels.numpy())  # Convert tensor to numpy array and flatten

# Count the frequency of each class
class_counts = Counter(all_labels)

# Get the classes and their corresponding counts
classes, counts = zip(*class_counts.items())
class_names = data.class_names
class_names = [class_names[i] for i in classes]
counts = counts / np.sum(counts) * 100 #convert to percent

# Sort by frequency
sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
sorted_classes = np.array(classes)[sorted_indices]
sorted_counts = np.array(counts)[sorted_indices]

sorted_class_names = [class_names[i] for i in sorted_indices]

# %% make a bar graph

classes_in_red = ['actinic keratosis', 'basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']

# Create a list of colors based on whether the class should be highlighted
colors = ['lightcoral' if cls in classes_in_red else 'skyblue' for cls in sorted_class_names]

# Plot the frequency of each class, sorted by frequency
plt.figure(figsize=(6, 3))
plt.bar(np.arange(len(classes)), sorted_counts, color=colors)
plt.xlabel('Class label')
plt.ylabel('Frequency (%)')
plt.title('Class frequency in dataset')
plt.xticks(np.arange(len(classes)))  # Show class labels on the x-axis

ax = plt.gca()
ax.set_xticklabels(sorted_class_names,ha='right')
plt.xticks(rotation=45)

# Remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create legend patches
red_patch = mpatches.Patch(color='lightcoral', label='Requires treatment')
blue_patch = mpatches.Patch(color='skyblue', label='Does not require treatment')

# Add the legend to the plot
plt.legend(handles=[red_patch, blue_patch])

# Save the figure to file
plt.savefig('./figures/Class_frequency_in_data' + set_name + '.png', bbox_inches='tight',transparent=True)

# Show the plot
plt.show()
