#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the input image through the neural network model
"""
import sys
from tensorflow import keras
from src.artist import InputImage, OutputImage


def predict(IMAGE_PATH):
    # Read in the passed image
    # Input is resized to 50x50 px
    img_in = InputImage(IMAGE_PATH)

    # Reshape to a 1-page batch and feed through model
    model_feed = img_in.data.reshape(1, 50, 50, 1)  # [x1, y1, x2, y2]
    model_out = model.predict(model_feed)

    # Concatenate model outputs to one array
    shapes_out = model_out[0, 0, :, :, :]

    # Filter to the number of shapes desired (NUM_SHAPES)
    print(NUM_SHAPES)
    shapes_filt = shapes_out[:NUM_SHAPES, :, :]

    # Print the numerical output to the console
    print(shapes_filt)
    shapes_out = str(shapes_filt)

    # Wrap the raw output in the OutputImage class
    img_out = OutputImage(100, 100, lines=shapes_filt)

    # Display both the input and output image
    img_in.display()
    img_out.display()

    return shapes_out


if (__name__ == '__main__'):

    try:
        IMAGE_PATH = sys.argv[1]
    except IndexError:
        IMAGE_PATH = input('Which image should I match?')

    NUM_SHAPES = 1
#    TODO Enable multi-shape output
#    try:
#        NUM_SHAPES = int(sys.argv[2])
#    except IndexError:
#        NUM_SHAPES = int(input('How many shapes do you want?'))

    try:
        MODEL_PATH = sys.argv[2]
    except IndexError:
        MODEL_PATH = input('Which saved model should I use?')

    model = keras.models.load_model(
        MODEL_PATH)

    predict(IMAGE_PATH)
