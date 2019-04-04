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
        NUM_SHAPES = input('How many shapes do you want?')

    MODEL_PATH = input('Which saved model should I use?')
    model = keras.models.load_model(
        MODEL_PATH
        )

    # Read in the passed image
    img_in = InputImage(IMAGE_PATH)

    # Reshape to a 1-page batch and feed through model
    model_feed = img_in.data.reshape(1, 512, 512, 1)  # TODO Allow sizes
    model_out = model.predict(model_feed)  # [2xpos, 2xsiz, 1xrot]

    # Concatenate model outputs to one array
    pred_pos = model_out[0][0, :, :]  # 30x2 position array
    pred_siz = model_out[1][0, :, :]  # 30x2 size array
    pred_rot = model_out[2][0, :].reshape(30, 1)  # 30x1 rotation array
    shapes_out = np.hstack((pred_pos, pred_siz, pred_rot))

    # Filter to the number of shapes desired (NUM_SHAPES)
    print(NUM_SHAPES)
    shapes_filt = shapes_out[:NUM_SHAPES, :]

    # Print the numerical output to the console
    print(shapes_filt)

    # Wrap the raw output in the OutputImage class
    img_out = OutputImage(512, 512, shapes_filt)  # TODO sizes

    # Display both the input and output image
    img_in.display()
    img_out.display()
