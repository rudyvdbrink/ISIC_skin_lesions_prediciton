# %% libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from PIL import Image

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

def prediction_barplot(counts):

    #define classes manually    
    #class_names = ['AK', 'BCC', 'DF', 'MLN', 'NV', 'PBK', 'SCC', 'VL']
    class_names = ['actinic keratosis', 
                   'basal cell carcinoma', 
                   'dermatofibroma',
                   'melanoma', 
                   'nevus',
                   'pigmented benign keratosis',
                   'squamous cell carcinoma',
                   'vascular lesion']

    #sort classes by probability
    sorted_indices     = np.argsort(counts)[::-1]  # Sort in descending order
    sorted_counts      = counts.transpose()[sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]    

    classes_in_red = ['actinic keratosis', 'basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']
    #classes_in_red = ['AK', 'BCC', 'MLN', 'SCC']

    #create a list of colors based on whether the class should be highlighted in red
    colors = ['lightcoral' if cls in classes_in_red else 'skyblue' for cls in sorted_class_names]

    #plot class probability
    fig = plt.figure(figsize=(6, 3))
    plt.bar(np.arange(len(class_names)), sorted_counts, color=colors)
    plt.xlabel('Class label')
    plt.ylabel('Probability (%)')
    plt.title('Predicted probability')
    plt.xticks(np.arange(len(class_names)))  # Show class labels on the x-axis

    #format plot
    ax = plt.gca()
    ax.set_xticklabels(sorted_class_names,ha='right')
    plt.xticks(rotation=45)   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    red_patch = mpatches.Patch(color='lightcoral', label='Requires treatment')
    blue_patch = mpatches.Patch(color='skyblue', label='Does not require treatment')
    plt.legend(handles=[red_patch, blue_patch])

    fig.patch.set_alpha(0.0)  #transparent background for the figure
    ax.patch.set_alpha(0.0)   #transparent background for the axes

    return fig

# %% model input / output handeling

#Preprocess the image to match the model input requirements
def preprocess_image(img, target_size):

    if img.mode != 'RGB':
        #convert the image to RGB if needed
        img = img.convert('RGB')

    #resize the image to the model's expected input shape
    img = img.resize((target_size[1], target_size[0]))

    #convert the image to a numpy array
    img = np.array(img, dtype=np.float32)

    #add a batch dimension since the model expects a batch of inputs
    img = np.expand_dims(img, axis=0)
       
    return img
    
#Compute percentage probabilities from raw model output
def rescale_to_probability(logits):

    #apply softmax function to raw scores (logits)
    percs = tf.nn.softmax(logits) 

    #convert to percentage
    percs *= 100

    return  percs.numpy()[0]


# %% TFlite model prediction

def make_tfl_prediction(model_name,img):

    #load the TFLite model and allocate tensors
    try:
        interpreter = tf.lite.Interpreter(model_path="../models/" + model_name)
    except:
        interpreter = tf.lite.Interpreter(model_path="./models/" + model_name)

    interpreter.allocate_tensors()

    #get input and output tensor details
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #preprocess the input image
    #image_path = "./example_imgs/0001_nevus.png"

    input_shape = input_details[0]['shape'][1:3]  # Get input shape (height, width)
    preprocessed_image = preprocess_image(img, input_shape)

    #set the model input
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

    #run inference
    interpreter.invoke()

    #get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data