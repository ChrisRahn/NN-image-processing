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
from artist import CustomImage, ImageBundle, InputImage, OutputImage
import pickle
import sys
from keras.layers import MaxPool2D, Conv2D, BatchNormalization, Dropout, Flatten, Dense


# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped

model_input = keras.layers.Input(shape=(512, 512, 1), data_format='channels_last')

Pool1 = MaxPool2D(pool_size=(2, 2))(model_input)  # (256, 256, 1)
Pool2 = MaxPool2D(pool_size=(2, 2))(Pool1)  # (128, 128, 1)
Pool3 = MaxPool2D(pool_size=(2, 2))(Pool2)  # (64, 64, 1)
Pool4 = MaxPool2D(pool_size=(2, 2))(Pool3)  # (32, 32, 1)
Pool5 = MaxPool2D(pool_size=(2, 2))(Pool4)  # (16, 16, 1)

Conv1 = Conv2D(5, (3, 3), padding='same')(Pool1)  # (256, 256, 5)
Conv2 = Conv2D(5, (3, 3), padding='same')(Pool2)  # (128, 128, 5)
Conv3 = Conv2D(5, (3, 3), padding='same')(Pool3)  # (64, 64, 5)
Conv4 = Conv2D(5, (3, 3), padding='same')(Pool4)  # (32, 32, 5)
Conv5 = Conv2D(5, (3, 3), padding='same')(Pool5)  # (16, 16, 5)

BN1 = BatchNormalization(axis=2)(Conv1)
BN2 = BatchNormalization(axis=2)(Conv2)
BN3 = BatchNormalization(axis=2)(Conv3)
BN4 = BatchNormalization(axis=2)(Conv4)
BN5 = BatchNormalization(axis=2)(Conv5)

Drop1 = Dropout(0.25)(BN1)
Drop2 = Dropout(0.25)(BN2)
Drop3 = Dropout(0.25)(BN3)
Drop4 = Dropout(0.25)(BN4)
Drop5 = Dropout(0.25)(BN5)

F1 = Flatten()(Drop1)  #
F2 = Flatten()(Drop2)
F3 = Flatten()(Drop3)
F4 = Flatten()(Drop4)
F5 = Flatten()(Drop5)

Dense1 = Dense(256)(F1)
Dense2 = Dense(256)(F2)
Dense3 = Dense(256)(F3)
Dense4 = Dense(256)(F4)
Dense5 = Dense(256)(F5)

Out1 = Dense(5, activation='relu')(Dense1)
Out2 = Dense(5, activation='relu')(Dense2)
Out3 = Dense(5, activation='relu')(Dense3)
Out4 = Dense(5, activation='relu')(Dense4)
Out5 = Dense(5, activation='relu')(Dense5)

model = keras.Model(inputs=model_input, outputs=[Out5, Out4, Out3, Out2, Out1])

# Define optimizer
optimizer = keras.optimizers.Adadelta()

# Define custom loss function

# ??? def scaled_mse(y_true, y_pred):
#    '''Loss b/w (30, 5) tensors ([x_pos, y_pos, w_scale, h_scale, rot])
#    x_pos & y_pos: [0, 512]; w_scale, h_scale: [0.1, 4]; rot: [0, 2*pi]
#    Scale all to be [0, 1]'''
#    y_pred = ops.convert_to_tensor(y_pred)
#    y_true = math_ops.cast(y_true, y_pred.dtype)
#    w = math_ops.cast(np.array([[1/512, 1/512, 1/4, 1/4, 1/(2*np.pi)]]), y_pred.dtype)
#    w = K.transpose(w)
#    return K.mean(math_ops.square(K.dot((y_true - y_pred),w)), axis=-1)

# Define losses
losses = {
    'Out5': 'mean_absolute_percentage_error',
    'Out4': 'mean_absolute_percentage_error',
    'Out3': 'mean_absolute_percentage_error',
    'Out2': 'mean_absolute_percentage_error',
    'Out1': 'mean_absolute_percentage_error'}

# Define loss weights
weights = {
    'Out5': 2.00,
    'Out4': 1.75,
    'Out3': 1.50,
    'Out2': 1.25,
    'Out1': 1.00}

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=weights,
    metrics=['mean_absolute_percentage_error'])

if (__name__ == '__main__'):
#    assert len(sys.argv) == 3, 'Pass me both the training and save filepaths!'
    # XXX Testing constants - Remove
    try:
        TRAINING_SET = sys.argv[1]
        SAVE_PATH = sys.argv[2]
    except IndexError:
        print('Pass me both the training set and save filepaths!')
        TRAINING_SET = '../data/train_set_04.pkl' # HINT input('What\'s the training set filepath?')
        TESTING_SET = '../data/test_set_04.pkl'
        SAVE_PATH = '../models/saved_model_04.h5' # HINT input('What\'s the saved model filepath?')
#        sys.exit()

    # Load the training set from the pickled ImageBundle
    train_bundle = pickle.load(open(TRAINING_SET, 'rb'))
    train_X = train_bundle.images
    train_y = train_bundle.tri_list

    # Load the testing set from the pickled ImageBundle
    test_bundle = pickle.load(open(TESTING_SET, 'rb'))
    test_X = test_bundle.images
    test_y = test_bundle.tri_list

    # IN: (samples, rows, cols, channels)
    IN_SHAPE = train_X.shape
    # OUT: (samples, shape_idx, shape_attrs, channels)
    OUT_SHAPE = train_y.shape
    # Initialize the training set

    # Fit the model to the training ImageBundle
    model.fit(
        train_X,
        train_y[:, :, :, 0],
        epochs=10,
        verbose=1,
        batch_size=20)

    # Write model config to YAML
    model_yaml = model.to_yaml()
    with open('../models/model_config.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)

    # Save model
    model.save(SAVE_PATH, overwrite=True, include_optimizer=True)
    print('\nModel saved at: %s' % SAVE_PATH)

    # Model evalutaion
    print(model.evaluate(
            test_X,
            test_y[:, :, :, 0]))
