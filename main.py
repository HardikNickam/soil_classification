from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint
# from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('best_model.hdf5')


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_image():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    img = image.load_img(image_path)
    img = img.resize((224, 224))
    # Convert the image to a numpy array and normalize the pixel values
    img_array = np.array(img) / 255.0
    # Add a batch dimension to the array
    img_array = np.expand_dims(img_array, axis=0)
    # Make the prediction
    pred = model.predict(img_array)
    # Get the true label
    true_label = np.argmax(pred)

    return render_template("index.html",prediction=true_label)


if __name__ == '__main__':
    app.run(debug=True)
