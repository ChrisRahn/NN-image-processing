# -*- coding: utf-8 -*-
'''
Serve the web app
'''

from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
from tensorflow import keras
import src

# Suppress TensorFlow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Create the app object that will route our calls
app = Flask(__name__)

# Load model
model = keras.models.load_model('./models/fracture.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    input_fp = req['file']
    print(input_fp)
    output = src.predictor.predict(input_fp)
    return jsonify(output)


if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=3333, debug=True)
