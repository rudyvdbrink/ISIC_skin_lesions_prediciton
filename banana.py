
# %%
import tensorflow as tf
import numpy as np
from PIL import Image

from supporting_functions import prediction_barplot

# %%
#Preprocess the image to match the model input requirements
def preprocess_image(img, target_size):

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


# %%
model_name = 'Xception_fair.tflite'

image_path = "./example_imgs/0001_nevus.png"
image_path = './example_imgs/test.png'

img = Image.open(image_path)
img = img.convert('RGB')

output_data    = make_tfl_prediction(model_name, img)
probabilities  = rescale_to_probability(output_data)

prediction_barplot(probabilities)

# %%

# model_name = 'Xception_fair.tflite'
# img = './example_imgs/test.png'
# img = './example_imgs/0001_nevus.png'

# img = Image.open(img)

# img = preprocess_image(img, (150, 200))

# pred = make_tfl_prediction(model_name,img)