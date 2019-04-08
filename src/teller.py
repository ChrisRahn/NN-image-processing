#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the input image through the neural network model
"""
import sys
from tensorflow import keras
from artist import InputImage, OutputImage
import numpy as np

if (__name__ == '__main__'):
    # ??? assert len(sys.argv) == 2, 'Gotta give me a JPEG to predict!'
    # XXX Testing constants - Remove
    try:
        IMAGE_PATH = sys.argv[1]
    except IndexError:
        IMAGE_PATH = input('Which image should I match?')

    try:
        NUM_SHAPES = int(sys.argv[2])
    except IndexError:
        NUM_SHAPES = int(input('How many shapes do you want?'))

    MODEL_PATH = input('Which saved model should I use?')
    model = keras.models.load_model(
        MODEL_PATH
        )

    # Read in the passed image
    img_in = InputImage(IMAGE_PATH)

    # Reshape to a 1-page batch and feed through model
    model_feed = img_in.data.reshape(1, 50, 50, 1)
    model_out = model.predict(model_feed)  # [x1, y1, x2, y2]

    # Concatenate model outputs to one array
    shapes_out = model_out[0, 0, :, :, :]

    # Filter to the number of shapes desired (NUM_SHAPES)
    print(NUM_SHAPES)
    shapes_filt = shapes_out[:NUM_SHAPES, :, :]

    # Print the numerical output to the console
    print(shapes_filt)

    # Wrap the raw output in the OutputImage class
    img_out = OutputImage(100, 100, lines=shapes_filt)

    # Display both the input and output image
    img_in.display()
    img_out.display()
