#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the neural network model
"""

from matplotlib.pyplot import imread, imshow
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from train_gen import simple_tri

#img = imread('../data/triforce.jpg')  # Grey-scale only for now
img = simple_tri()

# Rescale img's values to be [0 - 255]
# img = img * 255.  #TODO Remove prob

# Set training data to be random and test to be the image
train_X = simple_tri()
train_y = np.array([[5, 5, 11]])
# TensorFlow expects 5D tensors of kind (batch_size, rows, cols, channels)
IN_SHAPE = train_X.shape
OUT_SHAPE = train_y.shape
train_X = np.resize(train_X, (1, train_X.shape[0], train_X.shape[1], 1))

# Define a simple triangular kernel
tiny_tri = np.array([[0, 1, 0],
                     [0, 1, 0],
                     [1, 0, 1]])

triangle = np.array([[0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]])

triangle_2 = simple_tri()

kernel_t = tf.constant_initializer(tiny_tri)


def my_rescaler(X):
    max_val = K.max(K.max(X, axis=2, keepdims=True), axis=3)[0, 0, 0, 0]
    min_val = K.min(K.min(X, axis=2, keepdims=True), axis=3)[0, 0, 0, 0]
    return 255*(X - min_val)/(max_val - min_val)


model = keras.Sequential([
        # Maxpool the image
        keras.layers.MaxPool2D(
            input_shape=(IN_SHAPE[0], IN_SHAPE[1], 1),
            pool_size=(1, 1),
            padding='same',
            data_format='channels_last'),
        # Convolve the pooled image by the shape kernel(s)
        keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            activation='relu',
            # use_bias=True,
            kernel_initializer=kernel_t),
        # Basic Dense layer
#        keras.layers.Dense(
#            units=121,
#            activation='relu',
#            use_bias=True),
        # Output layer
        keras.layers.Dense(
            units=3,
            activation='relu',
            use_bias=True),
        #keras.layers.Flatten(),
        # Reshape down to the acutal features
#        keras.layers.Reshape(
#            target_shape=OUT_SHAPE)
        ])

# Compile the model
model.compile(optimizer='adadelta',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Fit the model to the training image
model.fit(train_X, train_y, epochs=100, verbose=1, batch_size=1)

# Evaluate model
# TODO model.evaluate(...)

# Repredict the training img
test_X = train_X
y_pred = model.predict(test_X)
imshow(test_X[0, :, :, 0])
print('Trained shape data: ', train_y)
print('Determined shape data: ', y_pred)
