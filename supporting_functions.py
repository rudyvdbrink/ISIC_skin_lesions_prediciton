# %% libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# %% data / labels retrieval

def retrieve_labels(ds):
    #get target from the dataset
    y = np.concatenate([y for _, y in ds], axis=0)
    return np.argmax(y, axis=1)

def retrieve_data(ds):
    #get data from the dataset
    x = np.concatenate([x for x, _ in ds], axis=0).astype('uint8')
    return x

# %% ploting

def plot_images_grid_nometa(images_array,start_N=0, num_images=12, grid_shape=(3, 4)):
    # Create a figure with a specified size
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 10))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for i in range(num_images):
        if i < len(images_array):
            # Plot each image
            axes[i].imshow(images_array[i+start_N])
            axes[i].axis('off')  # Hide the axis

    # Remove any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def evaluation_plots(model, ds):

    #get classes
    class_names = ds.class_names
    num_classes = len(class_names)

    #get true target from the dataset
    y = np.concatenate([y for _, y in ds], axis=0)
    y = np.argmax(y, axis=1)

    #make prediction
    y_pred_proba = model.predict(ds)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    labels = ['AK', 'BCC', 'DF', 'MLN', 'NV', 'PBK', 'SCC', 'VL']

    print(classification_report(y_pred,y,zero_division=0))

    print('Accuracy = ' + str(np.mean(y_pred==y)))
    print('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y,y_pred)))

    #create subplots
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    #make the subplots square
    axes[0].set_aspect('equal', 'box')
    axes[1].set_aspect('equal', 'box')

    #plot the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred,normalize='true')
    sns.heatmap(conf_matrix, annot=True, cmap="inferno", vmin=0, vmax=1, 
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Balanced accuracy = ' + str(metrics.balanced_accuracy_score(y,y_pred)))

    #plot the ROC curves for each class
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(to_categorical(y, num_classes)[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

    axes[1].plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')  #diagonal line
    axes[1].set_xlim([-0.05, 1.05])
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC curves per class')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

