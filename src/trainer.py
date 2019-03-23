#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the neural network model
"""

from matplotlib.pyplot import imread, imshow
import numpy as np
import tensorflow as tf
from tensorflow import keras

img = imread('../data/triforce.jpg')  # Grey-scale only for now

# Flatten img's values to be [0 - 1]
img = img / 255.

# Set training data to be random and test to be the image
train_X = np.random.random(size=img.shape)
train_y = img
# TensorFlow expects 5D tensors of kind (samples, time, channels, rows, cols)
train_X = np.resize(train_X, (260, 260, 0, 0, 0))
train_y = np.resize(train_y, (260, 260, 0, 0, 0))

# Define a simple triangular kernel
triangle = np.array([[0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]])

kernel_t = tf.constant_initializer(triangle)

# Define a simple LSTM model
model = keras.Sequential([
        keras.layers.ConvLSTM2D(
                 input_shape=(260, 260, 0, 0),
                 filters=2,
                 kernel_size=7,
                 strides=1,
                 dilation_rate=(1, 1),
                 activation='relu',
                 recurrent_activation='relu',
                 # kernel_initializer=kernel_t,
                 recurrent_initializer=None,
                 kernel_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 stateful=False)
        ])

# Compile the model
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Fit the model to the training image
model.fit(train_X, train_y, epochs=2, verbose=2)
