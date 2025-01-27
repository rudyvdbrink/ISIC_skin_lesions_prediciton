# %% libraries
import shap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from shap.plots.colors import red_white_blue
from supporting_functions import preprocess_image

# %%
def get_model_specs(model_name):
    #load the TFLite model and allocate tensors
    try:
        interpreter = tf.lite.Interpreter(model_path="../models/" + model_name)
    except:
        interpreter = tf.lite.Interpreter(model_path="./models/" + model_name)

    interpreter.allocate_tensors()

    #get input and output tensor details
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:3]  #get input shape (height, width)

    return input_shape, input_details, output_details, interpreter

# %% class for SHAP values computation

class TFLiteModel:
    def __init__(self, input_details, output_details, interpreter, preprocessed_image):
        
        self.input_details = input_details
        self.output_details = output_details
        self.interpreter = interpreter
        self.img = preprocessed_image

    def process(self, preprocessed_image):

        self.interpreter.allocate_tensors()  
        if preprocessed_image.shape[0] == 1: #single image
            #run model normally
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
            self.interpreter.invoke() #run inference        
            output_data = self.interpreter.get_tensor(self.output_details[0]['index']) #get the output
        else: #batched input
            #run model for each image in the batch sequentially
            output_data = []
            for i in range(preprocessed_image.shape[0]):
                self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image[i:i+1])
                self.interpreter.invoke()
                output_data.append(self.interpreter.get_tensor(self.output_details[0]['index']))
            output_data = np.squeeze(np.array(output_data))

            #this is the correct way to do batching with TFLite but for some reason it crashes the kernel
            # interpreter.resize_tensor_input(input_details[0]['index'],[preprocessed_image.shape[0], 150, 200, 3])
            # interpreter.allocate_tensors()
            # interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
            # # run the inference
            # interpreter.invoke()
            # output_data = interpreter.get_tensor(output_details[0]['index'])

        return output_data

# %% function for computing shap values

def compute_shap_values(model_name,img):

    # Definitions
    class_names = [ 'actinic keratosis', 
                'basal cell carcinoma', 
                'dermatofibroma',
                'melanoma', 
                'nevus',
                'pigmented benign keratosis',
                'squamous cell carcinoma',
                'vascular lesion']
    
    # Model setup
    input_shape, input_details, output_details, interpreter = get_model_specs(model_name)
    preprocessed_image = preprocess_image(img, input_shape)
    model = TFLiteModel(input_details, output_details, interpreter, preprocessed_image)
    f = model.process #function that takes in image input, and returns model output

    # Define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("inpaint_telea", (150, 200, 3))

    # By default the Partition explainer is used
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # Use 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(preprocessed_image, max_evals=500)

    return shap_values

# %% plotting


def plot_shap_values(shap_values, img, category_index=4):

    # Resize image
    preprocessed_image = preprocess_image(img, (150, 200))
    
    # Select the Shapley values for the desired category (e.g., category index 4)
    shap_values_category = np.sum(shap_values.values[0, :, :, :, category_index], axis=2)

    # Calculate the color limit
    max_abs_shap_value = np.max(np.abs(shap_values_category))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the image
    im = ax.imshow(np.squeeze(preprocessed_image/max(preprocessed_image.flatten())))
    ax.axis('off')

    # Overlay the heatmap of Shapley values with some transparency
    cmap = red_white_blue
    heatmap = ax.imshow(shap_values_category, cmap=cmap, alpha=0.3, vmin=-max_abs_shap_value, vmax=max_abs_shap_value)

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)

    # Append axes below the image axes with a smaller pad value
    cax = divider.append_axes("bottom", size="5%", pad=0.1)

    # Create the colorbar
    cbar = plt.colorbar(heatmap, cax=cax, orientation='horizontal')
    cbar.outline.set_visible(False)

    # Set colorbar ticks to show only min, max, and 0 values
    cbar.set_ticks([-max_abs_shap_value, 0, max_abs_shap_value])
    cbar.set_ticklabels([f'{-max_abs_shap_value:.2f}', '0', f'{max_abs_shap_value:.2f}'])
    cbar.set_label('SHAP values')

    # # Show the plot
    # plt.show()# %%

    return fig, ax