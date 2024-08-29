# %% libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# %% data loading

def load_dataset(data_dir):

    #parameters
    batch_size        = 256 #large batch size because we have a very imbalanced dataset
    img_width         = 600 #original image shape
    img_height        = 450 #original image shape
    rn_seed           = 42 #what random number seed to use

    #get .jpg data
    train_ds, test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        labels='inferred',
        label_mode='categorical',
        subset="both",
        seed=rn_seed,
        image_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True
    )

    return train_ds, test_ds


# %% ploting

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
