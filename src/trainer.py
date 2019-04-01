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
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import kernels
from artist import CustomImage, ImageBundle
import pickle
import sys

# Define a simple triangular kernel and kernel constraints
kernel_tri = tf.constant_initializer(kernels.triangle_5())
kernel_const = min_max_norm(0.001, None, rate=1, axis=0)
kernel_nonneg = non_neg()

# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped
model = keras.Sequential([
        # Maxpool the image
        keras.layers.MaxPool2D(
            input_shape=(512, 512, 1),
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
            units=5 * 30,  # !!! 30 output shapes per channel
            activation=None,
            # kernel_constraint=kernel_nonneg,
            use_bias=True),

        # Activation layer
        keras.layers.PReLU(),

        # Reshape & output
        keras.layers.Reshape((30, 5))
        ])

# Define optimizer
optimizer = keras.optimizers.Adadelta()

# Define custom loss function


def scaled_mse(y_true, y_pred):
    '''Loss b/w (30, 5) tensors ([x_pos, y_pos, w_scale, h_scale, rot])
    x_pos & y_pos: [0, 512]; w_scale, h_scale: [0.1, 4]; rot: [0, 2*pi]
    Scale all to be [0, 1]'''
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    scales = np.array([[1/512, 1/512, 1/4, 1/4, 1/(2*np.pi)]])
    y_true = y_true * scales
    y_pred = y_pred * scales
    return K.mean(math_ops.square(y_true - y_pred), axis=-1)


# Add the custom loss to the loss dictionary
tf.losses.add_loss(scaled_mse)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=scaled_mse,
    metrics=[scaled_mse])


if (__name__ == '__main__'):
#    assert len(sys.argv) == 3, 'Pass me both the training and save filepaths!'
    # XXX Testing constants - Remove
    try:
        TRAINING_SET = sys.argv[1]
        SAVE_PATH = sys.argv[2]
    except IndexError:
        print('Pass me both the training set and save filepaths!')
        TRAINING_SET = input('What\'s the training set filepath?')
        SAVE_PATH = input('What\'s the saved model filepath?')
#        sys.exit()

    # Load the training set from the pickled ImageBundle
    train_bundle = pickle.load(open(TRAINING_SET, 'rb'))
    train_X = train_bundle.images
    train_y = train_bundle.tri_list

    # IN: (samples, rows, cols, channels)
    IN_SHAPE = train_X.shape
    # OUT: (samples, shape_idx, shape_attrs, channels)
    OUT_SHAPE = train_y.shape
    # Initialize the training set

    # Fit the model to the training ImageBundle
    model.fit(
        train_X,
        train_y[:, :, :, 0],
        epochs=50,
        verbose=1,
        batch_size=5)

    # Write model config to YAML
    model_yaml = model.to_yaml()
    with open('../models/model_config.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)

    # Save model
    model.save(SAVE_PATH, overwrite=True, include_optimizer=True)
    print('\nModel saved at: %s' % SAVE_PATH)
