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
from artist import CustomImage, ImageBundle
import pickle
import sys

TRAINING_SET = '../data/train_set_01.pkl'
SAVE_FILEPATH = '../checkpoints/fracture.ckpt'

# Load the training set from the pickled ImageBundle
train_bundle = pickle.load(open(TRAINING_SET, 'rb'))
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

        # Activation layer
        keras.layers.PReLU(),

        # Reshape & output
        keras.layers.Reshape((5, 5))
        ])

# Compile the model
model.compile(optimizer='adadelta',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# ??? Define model logging
logger = keras.callbacks.ModelCheckpoint(
        '../checkpoints/model_01.ckpt',
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='auto',
        period=25)
logger.model = model

# Define model saver
saver = tf.train.Saver(
        var_list={'frac_model': model},
        reshape=False,
        max_to_keep=5,
        # filename=SAVE_FILEPATH
        )

if (__name__ == '__main__'):
    with tf.Session() as sess:
        # Initialize the training set

        # TODO sess.run(initializer)

        # Fit the model to the training ImageBundle
        model.fit(
            train_X,
            train_y[:, :, :, 0],
            epochs=50,
            verbose=1,
            batch_size=5)

        # Save session checkpoint
        save_path = saver.save(sess, SAVE_FILEPATH)
        print('Model saved at: %s' % save_path)
