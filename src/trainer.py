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
from tensorflow.keras.constraints import min_max_norm, non_neg
import kernels
from train_gen import CustomImage, ImageBundle
import pickle

# Load the training set from the pickled ImageBundle
train_bundle = pickle.load(open('../data/train_set_01.pkl', 'rb'))
train_X = train_bundle.images
train_y = train_bundle.tri_list

# IN: (samples, rows, cols, channels)
IN_SHAPE = train_X.shape
# OUT: (samples, shape_idx, shape_attrs, channels)
OUT_SHAPE = train_y.shape

# Define a simple triangular kernel and kernel constraints
kernel_tri = tf.constant_initializer(kernels.triangle_5())
kernel_const = min_max_norm(0.001, None, rate=1, axis=0)
kernel_nonneg = non_neg()

# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped
model = keras.Sequential([
        # Input layer
        keras.layers.InputLayer(
            input_shape=(512, 512, 1)),

        # Maxpool the image
        keras.layers.MaxPool2D(
            # input_shape=(512, 512, 1),
            pool_size=2,
            padding='same',
            data_format='channels_last'),

        # Convolve the pooled image by the shape kernel(s)
        # ??? Use LocallyConnected2D instead?
        keras.layers.Conv2D(
            filters=5,
            kernel_size=(8, 8),
            strides=(8, 8),
            padding='same',
            data_format='channels_last',
            activation='sigmoid',
            use_bias=True),
            # ??? kernel_initializer=kernel_tri,
            # kernel_constraint=kernel_nonneg),
        keras.layers.Conv2D(
            filters=5,
            kernel_size=(8, 8),
            strides=(8, 8),
            padding='same',
            data_format='channels_last',
            activation='sigmoid',
            use_bias=True),
        # Flatten
        keras.layers.Flatten(),

        # Basic Dense layer
        keras.layers.Dense(
            units=25,
            activation=None,
            # kernel_constraint=kernel_nonneg,
            use_bias=True),

        # Another Dense layer
#        keras.layers.Dense(
#            units=5,
#            activation=None,
#            # kernel_constraint=kernel_nonneg,
#            use_bias=True),

        # Output layer
        keras.layers.PReLU(),
        
        # Reshape output
        keras.layers.Reshape((5,5))
        ])

# Compile the model
model.compile(optimizer='adadelta',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Fit the model to the training image
model.fit(train_X, train_y[:,:,:,0], epochs=500, verbose=1, batch_size=5)

# Evaluate model
# TODO model.evaluate(...)

# Repredict the training img
#test_X = 255 - imread('../data/triforce.jpg')
#test_X = np.reshape(test_X, (1, test_X.shape[0], test_X.shape[1], 1))
#y_pred = model.predict(test_X)
#imshow(test_X[0, :, :, 0])
#print('Trained shape data: ', train_y)
#print('Determined shape data: ', y_pred)
#y_pred
