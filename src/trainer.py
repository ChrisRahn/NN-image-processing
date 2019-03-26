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
train_X = np.resize(train_X, (1, train_X.shape[0], train_X.shape[1], 1))
IN_SHAPE = train_X.shape
OUT_SHAPE = train_y.shape
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

kernel_t = tf.constant_initializer(triangle_2)


def my_rescaler(X):
    max_val = K.max(K.max(X, axis=2, keepdims=True), axis=3)[0, 0, 0, 0]
    min_val = K.min(K.min(X, axis=2, keepdims=True), axis=3)[0, 0, 0, 0]
    return 255*(X - min_val)/(max_val - min_val)


model = keras.Sequential([
        # Maxpool the image
        keras.layers.MaxPool2D(
            input_shape=(train_X.shape[1], train_X.shape[2], 1),
            pool_size=(26, 26),
            padding='same',
            data_format='channels_last'),
        # Convolve the pooled image by the shape kernel(s)
        keras.layers.Conv2D(
            filters=1,
            kernel_size=(11, 11),
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            activation='relu',
            # use_bias=True,
            kernel_initializer=kernel_t),
        # Basic Dense layer
        keras.layers.Dense(
            units=3,
            activation='relu',
            use_bias=True),
        # Reshape down to the acutal features
        keras.layers.Reshape(
            target_shape=train_y.shape)
        ])

# Compile the model
model.compile(optimizer='adadelta',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Fit the model to the training image
model.fit(train_X, train_y, epochs=5, verbose=2, batch_size=1)

# Evaluate model
# TODO model.evaluate(...)

## Predict a random array for testing
#test_X = np.resize(np.random.random(size=img.shape), (1, 1, img.shape[0], img.shape[1], 1))
#pred_y = model.predict(test_X)
#pred_y_img = pred_y[0, 0, :, :, :]
#imshow(pred_y_img[:, :, 0])
#print('Mean values are: ', pred_y_img.mean(axis=0).mean(axis=0))
