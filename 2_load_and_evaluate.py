# %% libraries

import tensorflow as tf
import keras

import numpy as np

from supporting_functions import evaluation_plots, retrieve_labels, retrieve_data
from loading_functions import make_full_dataset_from_image_directory, make_balanced_split_dataset_from_image_directory

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

# %% definitions

#model_name = 'VGG19_multi-class_classifier_pretrained' #what model to evaluate
model_name = 'VGG19_multi-class_classifier_fully_trained' #what model to evaluate

data_dir   = './data/processed/HAM10000/' # where did we store the images

# %% load data

batch_size     = 32
validation     = True
shuffle        = True
split          = [0.7, 0.1, 0.2] #train, validation, test split proportions

train_ds, _, test_ds = make_balanced_split_dataset_from_image_directory(data_dir, 
                                                                 batch_size, 
                                                                 None, 
                                                                 split=split, 
                                                                 validation=validation, 
                                                                 shuffle=shuffle)

# %% load model

model = keras.saving.load_model("./models/" + model_name + ".keras")

# %% show training performance

evaluation_plots(model, train_ds)

# %% show testing performance

evaluation_plots(model, test_ds)

# %% retrieve data and labels for error analysis

x = retrieve_data(test_ds)
y = retrieve_labels(test_ds)
pred = model.predict(test_ds)
y_pred = np.argmax(pred,axis=1)

# %% function for row plotting images

def row_plot(selected_images, rows=1, cols=4):
   
    #shuffle for random selection
    np.random.shuffle(selected_images)

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Flatten the axes array to easily iterate through it
    axes = axes.flatten()

    # Plot images in the grid
    for i in range(rows * cols):
        if i < len(selected_images):
            # Plot the i-th selected image
            axes[i].imshow(selected_images[i].astype(np.uint8))
            axes[i].axis('off')  # Hide axis
        else:
            # Turn off unused axes if there are more grid spaces than images
            axes[i].axis('off')

    # Show the grid of images
    plt.tight_layout()
    fname = str(np.random.randint(0,10000) ) + '.png'
    plt.savefig('./figures/' + fname, transparent=True)

    plt.show()


# %% find and plot some actinic keratoses that it got wrong

# find some actinic keratoses that it thought were pigmented keratoses
ak_ind = y == 0
ak_pred = y_pred == 5

wrong_pred_ak = np.logical_and(ak_ind, ak_pred)

# find some pigmented keratoses that it got right
pbk_ind = y == 5
pbk_pred = y_pred == 5

right_pred_pbk = np.logical_and(pbk_ind, pbk_pred)

# find some actinic keratoses that it got right for comparison
right_pred_ak = np.logical_and(ak_ind,  y_pred == 0)

# %% plot

row_plot( x[wrong_pred_ak] )
row_plot( x[right_pred_pbk] )

# %% compute confidence

from supporting_functions import rescale_to_probability

prob_wrong_ak = model.predict(x[wrong_pred_ak])
confidence = tf.nn.softmax(prob_wrong_ak) * 100
confidence = confidence.numpy()
wrong_ak_conf = np.max(confidence,axis=1)

prob_right_ak = model.predict(x[right_pred_ak])
confidence = tf.nn.softmax(prob_right_ak) * 100
confidence = confidence.numpy()
right_ak_conf = np.max(confidence,axis=1)

# Compute mean and standard deviation
mean_values = np.array([np.mean(right_ak_conf), np.mean(wrong_ak_conf)])
std_values = np.array([np.std(right_ak_conf, ddof=1), np.std(wrong_ak_conf, ddof=1)])

# Define the categories
categories = ['Right', 'Wrong']

# Plotting the bar graph with error bars
plt.figure(figsize=(2, 3))
bars = plt.bar(categories, mean_values, yerr=std_values, capsize=5, color=['skyblue', 'lightcoral'])

# Add labels and title
plt.ylabel('Model confidence (%)')
plt.title('Actinic keratoses')
# Add labels and title
ax = plt.gca()
# ax.set_xticklabels(categories, rotation=45)

plt.ylim(0, 100)
# Remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.ylim(0, 110)
plt.savefig('./figures/AK_confidence.png', transparent=True,bbox_inches='tight')

# %% do the same but with melanoma / nevi

# find some melanoma that it thought were nevi
mln_ind = y == 3
mln_pred = y_pred == 4

wrong_pred_mln = np.logical_and(mln_ind, mln_pred)

# find some nevi that it got right
nv_ind = y == 4
nv_pred = y_pred == 4

right_pred_nv = np.logical_and(nv_ind, nv_pred)

right_pred_mln = np.logical_and(nv_ind,  y_pred == 4)


# %% plot

row_plot( x[wrong_pred_mln] )
row_plot( x[right_pred_nv] )

# %% compute confidence

from supporting_functions import rescale_to_probability

prob_wrong_mln = model.predict(x[wrong_pred_mln])
confidence = tf.nn.softmax(prob_wrong_mln) * 100
confidence = confidence.numpy()
wrong_mln_conf = np.max(confidence,axis=1)

prob_right_mln = model.predict(x[right_pred_mln])
confidence = tf.nn.softmax(prob_right_mln) * 100
confidence = confidence.numpy()
right_mln_conf = np.max(confidence,axis=1)

# Compute mean and standard deviation
mean_values = np.array([np.mean(right_mln_conf), np.mean(wrong_mln_conf)])
std_values = np.array([np.std(right_mln_conf, ddof=1), np.std(wrong_mln_conf, ddof=1)])

# Define the categories
categories = ['Right', 'Wrong']

# Plotting the bar graph with error bars
plt.figure(figsize=(2, 3))
bars = plt.bar(categories, mean_values, yerr=std_values, capsize=5, color=['skyblue', 'lightcoral'])

# Add labels and title
plt.ylabel('Model confidence (%)')
plt.title('Melanoma')
# Add labels and title
ax = plt.gca()
# ax.set_xticklabels(categories, rotation=45)

plt.ylim(0, 100)
# Remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.ylim(0, 110)
plt.savefig('./figures/MLN_confidence.png', transparent=True,bbox_inches='tight')



# %% recall plot

# Compute recall per class
recall_per_class = recall_score(y, y_pred, average=None)

# Compute average recall
average_recall = recall_score(y, y_pred, average='macro')

# Sort the recall values for individual classes
sorted_indices = np.argsort(recall_per_class)
sorted_indices = sorted_indices[::-1]

# Create labels for the classes
class_labels = ['Actinic keratosis', 
                'Basal cell carcinoma', 
                'Dermatofibroma',
                'Melanoma', 
                'Nevus',
                'Pigmented benign keratosis',
                'Squamous cell carcinoma',
                'Vascular lesion']


# Sort the recall values and labels (excluding the average recall)
sorted_recalls = recall_per_class[sorted_indices]
sorted_labels = np.array(class_labels)[sorted_indices]

# Insert the average recall at the beginning of both recall values and labels
sorted_recalls = np.concatenate([[average_recall], sorted_recalls])
sorted_labels = ['Average'] + sorted_labels.tolist()

# Plot the sorted bar plot
fig = plt.figure(figsize=(6, 3))
plt.bar(sorted_labels, sorted_recalls, color='skyblue')
plt.ylabel('Recall')
plt.title('Recall per class')


# Show the plot
ax = plt.gca()
ax.set_xticklabels(sorted_labels,ha='right')
ax.set_ylim([0, 1])
plt.xticks(rotation=45)   
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('./figures/Recall_' + model_name, transparent=True,bbox_inches='tight')

plt.show()

# %% confusion matrix

labels = ['AK', 'BCC', 'DF', 'MLN', 'NV', 'PBK', 'SCC', 'VL']
num_classes = len(labels)

font = {'size': 10,
'weight' : 'normal',}
matplotlib.rc('font', **font)

# Create a single plot
plt.figure(figsize=(6, 5))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y, y_pred, normalize='true') * 100
sns.heatmap(conf_matrix, annot=True, cmap="inferno", vmin=0, vmax=100, 
            xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Classification frequency (%)'})

# Set axis labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save the figure
plt.savefig('./figures/confusion_matrix.png', transparent=True)

# %%

from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical

roc_auc = np.empty(num_classes)
#get the ROC curves for each class
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(to_categorical(y, num_classes)[:, i], pred[:, i])
    roc_auc[i] = auc(fpr, tpr)

# %% plot of humans

import matplotlib.pyplot as plt
import numpy as np

# Define the data
categories = ['All', 'Consensus']
accuracies = [64.8, 73.7]
ci_lower = [62.4, 70.9]
ci_upper = [67.3, 76.6]

# Calculate the error bars as half the range of the confidence interval
errors = [np.array(ci_upper) - np.array(accuracies), np.array(accuracies) - np.array(ci_lower)]

# Plotting the bar graph with error bars
plt.figure(figsize=(2, 3))
bars = plt.bar(categories, accuracies, yerr=errors, capsize=5, color='skyblue')

# Add labels and title
ax = plt.gca()
ax.set_xticklabels(['Average', 'Consensus'], rotation=45)
# Add labels and title
plt.xlabel('Reader type')
plt.ylabel('Balanced accuracy (%)')
plt.title('Human performance')
plt.ylim(0, 100)
# Remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.ylim(0, 100)
plt.savefig('./figures/humans.png', transparent=True,bbox_inches='tight')

plt.show()


# %% plot of humans plus model

import matplotlib.pyplot as plt
import numpy as np

# Define the data
categories = ['All', 'Consensus']
accuracies = [64.8, 73.7]
ci_lower = [62.4, 70.9]
ci_upper = [67.3, 76.6]

# Calculate the error bars as half the range of the confidence interval
errors = [np.array(ci_upper) - np.array(accuracies), np.array(accuracies) - np.array(ci_lower)]

# Plotting the bar graph with error bars
plt.figure(figsize=(2, 3))
bars = plt.bar(categories, accuracies, yerr=errors, capsize=5, color='skyblue')

# Add labels and title
ax = plt.gca()
ax.set_xticklabels(['Average', 'Consensus'], rotation=45)
plt.plot([-0.5, 1.5], [82, 82],'k--')
plt.text(-0.5, 85,'Our model')

# Add labels and title
plt.xlabel('Reader type')
plt.ylabel('Balanced accuracy (%)')
#plt.title('Human performance')
plt.ylim(0, 100)
# Remove the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.ylim(0, 100)
plt.savefig('./figures/humans_plus_model.png', transparent=True,bbox_inches='tight')

plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt

# Example variables
# pred: model output, size (cases, classes)
# y: true labels, size (cases)

def plot_confidence_histograms(pred, y, bins=10):
    # Get the predicted class and confidence (max of softmax output)
    pred = tf.nn.softmax(pred) * 100
    predicted_class = np.argmax(pred, axis=1)
    confidence = np.max(pred, axis=1)
    
    # Correctly and incorrectly classified
    correct_conf = confidence[predicted_class == y]
    incorrect_conf = confidence[predicted_class != y]
    
    # Number of total cases
    total_cases = len(y)
    
    # Plotting the histograms
    plt.figure(figsize=(5, 3))
    
    # Create histograms with percentage scaling (density=True for percentages)
    plt.hist(correct_conf, bins=bins, alpha=0.5, label='Correct', density=True)
    plt.hist(incorrect_conf, bins=bins, alpha=0.5, label='Incorrect', density=True)
    
    # Add labels and title
    plt.xlabel('Model confidence')
    plt.ylabel('Percentage of total cases')
    plt.title('Model confidence per classifiation type')
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Display the plot
    plt.show()

# Pre-specify 10 bins between 0 and 100
bins = np.linspace(0, 100, 20)

#make plot usage
plot_confidence_histograms(pred, y, bins)


# %% plot humans

# import pandas as pd

# df = pd.read_csv('C:\DATA\ISIC_skin_lesions_prediciton\data\human\human_performance.csv')

# # Calculate Balanced Accuracy: (Sensitivity + Specificity) / 2
# df['Balanced Accuracy'] = (df['Sensitivity'].str.rstrip('%').astype(float) + 
#                            df['Specificity'].str.rstrip('%').astype(float)) / 2

# # Group by Reader Type and calculate mean and standard deviation for error bars
# grouped = df.groupby('Reader Type')['Balanced Accuracy'].agg(['mean', 'std'])

# # Plotting the bar graph with error bars
# plt.figure(figsize=(2, 3))
# plt.bar(grouped.index, grouped['mean'], yerr=grouped['std'], capsize=5, color='skyblue')

# ax = plt.gca()
# ax.set_xticklabels(['All', 'Experts'], rotation=45)
# # Add labels and title
# plt.xlabel('Reader type')
# plt.ylabel('Balanced accuracy (%)')
# plt.title('Balanced accuracy')
# plt.ylim(0, 100)
# # Remove the upper and right spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)


# # Show the plot
# plt.show()


# %% examine accuracy on the binary problem

# y = retrieve_labels(test_ds)
# y = np.where(np.isin(y, [0, 1, 3, 6]), 1, y)
# y = np.where(np.isin(y, [2, 4, 5, 7]), 0, y)

# pred = model.predict(test_ds)
# y_pred = np.argmax(pred,axis=1)
# y_pred = np.where(np.isin(y_pred, [0, 1, 3, 6]), 1, y_pred)
# y_pred = np.where(np.isin(y_pred, [2, 4, 5, 7]), 0, y_pred)

# print('Binary accraucy = ' + str(sum(y_pred == y)/len(y)))
# print('Binary balanced accraucy = ' + str(balanced_accuracy_score(y,y_pred)))

# cfm = confusion_matrix(y,y_pred,normalize='true')
# sns.heatmap(cfm,annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('True')

# %% examine accuracy without the 8th class

# y      = retrieve_labels(test_ds)
# y_pred = np.argmax(pred,axis=1)

# idx = np.where(y == 0)[0]

# y      = np.delete(y, idx)
# y_pred = np.delete(y_pred, idx)

# print('Class-corrected accraucy = ' + str(sum(y_pred == y)/len(y)))
# print('Class-corrected balanced accraucy = ' + str(balanced_accuracy_score(y,y_pred)))