import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras

# First we should load our model

model = tf.keras.models.load_model('saved_model/transfer-model')

# We can now load a list of class names from a file

file = open("class_names", "rb")
class_names = np.load(file)

# Url of a image we will be analysing

url = "https://images.dog.ceo/breeds/chihuahua/n02085620_1298.jpg"

# Let's give the image an unique id,
# and preprocess it

id = str(uuid.uuid1())

path = tf.keras.utils.get_file(id, origin=url)

img = keras.preprocessing.image.load_img(
    path, target_size=(160, 160)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# And we can finally make a prediction

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
