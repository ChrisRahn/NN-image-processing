#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the neural network model
"""

from matplotlib.pyplot import imread, imshow
import tensorflow as tf
from tensorflow import keras
from tensorflow import initializers

img = imread('../data/triforce.jpg')  # Grey-scale only for now

# Flatten img's values to be [0 - 255]
img = img / 255.

# Define a simple kernel
kernel_t = tf.constant_initializer

# Define a simple LSTM model
model = keras.Sequential([
        keras.layers.ConvLSTM2D(
                 filters=2,
                 kernel_size=13,
                 strides=1,
                 dilation_rate=(1, 1),
                 activation='relu',
                 recurrent_activation='relu',
                 kernel_initializer=kernel_t,
                 recurrent_initializer=None,
                 kernel_constraint=None,
                 return_sequences=False,
                 stateful=False)
        ])

# Compile the model
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
