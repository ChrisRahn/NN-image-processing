#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the input image through the neural network model
"""
import sys
from tensorflow import keras
from artist import InputImage, OutputImage

model = keras.models.load_model('../models/saved_model_01.h5')

if (__name__ == '__main__'):
    # ??? assert len(sys.argv) == 2, 'Gotta give me a JPEG to predict!'
    # XXX Testing constants - Remove
    try:
        IMAGE_PATH = sys.argv[1]
    except IndexError:
        IMAGE_PATH = '../data/triforce.jpg'

    # Read in the passed image
    img_in = InputImage(IMAGE_PATH)

    # Reshape to a 1-page batch and feed through model
    model_feed = img_in.data.reshape(1, 512, 512, 1)  # TODO Allow sizes
    shapes_out = model.predict(model_feed)[0, :, :]

    # Wrap the raw output in the OutputImage class
    img_out = OutputImage(512, 512, shapes_out)  # TODO sizes

    # Display both the input and output image
    img_in.display()
    img_out.display()
