# %% libraries

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from supporting_functions import preprocess_image, rescale_to_probability, make_tfl_prediction, prediction_barplot

# %%
model_name = 'Xception_multi-class_classifier_fully_trained_3rounds_quantized.tflite'

image_path = "./example_imgs/0001_nevus.png"
img = Image.open(image_path)

output_data    = make_tfl_prediction(model_name, img)
probabilities  = rescale_to_probability(output_data)

prediction_barplot(probabilities)


# %%

# # Step 1: Load the TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path="./models/" + model_name)
# interpreter.allocate_tensors()

# # Step 2: Get input and output tensor details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # %% match input

# # %%
# # Step 4: Preprocess the input image

# input_shape = input_details[0]['shape'][1:3]  # Get input shape (height, width)
# preprocessed_image = preprocess_image(image_path, input_shape)

# # %%
# # Step 5: Set the model input
# interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

# # Step 6: Run inference
# interpreter.invoke()

# # Step 7: Get the output
# output_data = interpreter.get_tensor(output_details[0]['index'])

# # Step 8: Postprocess the output (optional)
# # For example, if it's a classification model, you might want to interpret the output
# predicted_class = np.argmax(output_data)
# probabilities  = rescale_to_probability(output_data)

# confidence = np.max(probabilities)

# print(f"Predicted class: {predicted_class}, Confidence: {confidence}")


# # %% 

# prediction_barplot(probabilities)

# %%

# class_names = ['AK', 'BCC', 'DF', 'MLN', 'NV', 'PBK', 'SCC', 'VL']
# classes     = np.arange(len(class_names))

# counts = probabilities[0] 


# # Sort by frequency
# sorted_indices     = np.argsort(counts)[::-1]  # Sort in descending order
# sorted_counts      = counts.transpose()[sorted_indices]
# sorted_class_names = [class_names[i] for i in sorted_indices]

# # %% make a bar graph

# #classes_in_red = ['actinic keratosis', 'basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']
# classes_in_red = ['AK', 'BCC', 'MLN', 'SCC']

# # Create a list of colors based on whether the class should be highlighted
# colors = ['lightcoral' if cls in classes_in_red else 'skyblue' for cls in sorted_class_names]

# # Plot the frequency of each class, sorted by frequency
# plt.figure(figsize=(6, 3))
# plt.bar(np.arange(len(classes)), sorted_counts, color=colors)
# plt.xlabel('Class label')
# plt.ylabel('Probability (%)')
# plt.title('Predicted probability')
# plt.xticks(np.arange(len(classes)))  # Show class labels on the x-axis

# ax = plt.gca()
# ax.set_xticklabels(sorted_class_names,ha='right')
# plt.xticks(rotation=45)

# # Remove the upper and right spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Create legend patches
# red_patch = mpatches.Patch(color='lightcoral', label='Requires treatment')
# blue_patch = mpatches.Patch(color='skyblue', label='Does not require treatment')

# # Add the legend to the plot
# plt.legend(handles=[red_patch, blue_patch])

# # Save the figure to file
# #plt.savefig('./figures/Class_frequency_in_data' + set_name + '.png', bbox_inches='tight',transparent=True)

# # Show the plot
# plt.show()
