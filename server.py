# -*- coding: utf-8 -*-
'''
Serve the web app
'''

from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
from tensorflow import get_default_graph, Session
from tensorflow import keras
import numpy as np
import os
from src.predictor import predict as __predict
from src.artist import InputImage, OutputImage

# Suppress TensorFlow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Create the app object that will route our calls
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Fetch contents of /static directory
    img_list = [i for i in os.listdir('./static/')
                if i.split('.')[1] in ['png', 'jpg', 'jpeg']]

    return render_template('home.html', img_list=img_list)


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    print(req)
    input_fp = './static/segment.jpg'  # + req
    print(input_fp)
    mySession = Session()
    with mySession:
        model = keras.models.load_model('./models/fracture.h5')
        print(model.output)
        shapes_out = __predict(model, './static/segment.jpg')
    x1, y1, x2, y2 = np.round(shapes_out[0, :, 0], 2)
    out_str = '(' + str(x1) + ', ' + str(y1) + ')---(' \
        + str(x2) + ', ' + str(y2) + ')'
    return jsonify(out_str)


if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=3333, debug=True)
