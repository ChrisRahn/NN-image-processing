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

# Rescale img's values to be [0 - 1]
img = img / 255.

# Set training data to be random and test to be the image
train_X = np.random.random(size=img.shape)
train_y = img
# TensorFlow expects 5D tensors of kind (samples, time, rows, cols, channels)
train_X = np.resize(train_X, (1, 1, 260, 260, 1))
train_y = np.resize(train_y, (1, 1, 260, 260, 1))

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
                 input_shape=(None, 260, 260, 1),
                 filters=7,
                 kernel_size=(7, 7),
                 padding='same',
#                 strides=1,
#                 dilation_rate=(1, 1),
                 activation='relu',
#                 recurrent_activation='relu',
#                 kernel_initializer=kernel_t,
#                 recurrent_initializer=None,
#                 kernel_constraint=None,
                 return_sequences=True,
#                 return_state=False,
#                 stateful=False,
                 data_format='channels_last'),
# HINT        keras.layers.BatchNormalization()
        ])

# Compile the model
model.compile(optimizer='adadelta',  # ???Why Adadelta?
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Fit the model to the training image
model.fit(train_X, train_y, epochs=2, verbose=2, batch_size=1)

# Evaluate model
# TODO model.evaluate(...)

# Predict a random array for testing
test_X = np.resize(np.random.random(size=img.shape), (1, 1, 260, 260, 1))
pred_y = model.predict(test_X)
pred_y_img = pred_y[0, 0, :, :, :]
imshow(pred_y_img[:, :, 0])
