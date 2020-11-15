import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras

from flask import jsonify


def get_model_api():

    # First we should load the model

    model = tf.keras.models.load_model('../saved_model/transfer-model')

    # Now we can load class names

    file = open("class_names", "rb")
    class_names = np.load(file)

    def model_api(url):

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

        # Clean predicted class name
        # e.g. convert 'n02099601-golden_retriever' to 'Golden retriever'
        name = class_names[np.argmax(score)]
        name = name.split('-')[1].replace('_', ' ')
        # make first letter capital
        prediction_name = name[0].upper() + name[1:]

        return jsonify(
            url=url,
            prediction=str(np.argmax(score)),
            prediction_name=prediction_name,
            score=str(np.max(score))
        )

    return model_api
