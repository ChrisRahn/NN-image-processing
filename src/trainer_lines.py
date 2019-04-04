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
from kernels import sobel_x
from artist import CustomImage, ImageBundle, InputImage, OutputImage
import pickle
import sys
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, BatchNormalization, Dropout, Flatten, Concatenate, Dense, Reshape, Activation, Lambda, LeakyReLU

# Define Sobel filter
sobel_x = tf.constant_initializer(sobel_x())

# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped

# Define the model's layers
model_input = Input(shape=(512, 512, 1))

Lambda_In = Lambda(lambda x: x/255.)(model_input)

Pool = MaxPool2D(pool_size=(2, 2))(Lambda_In)  # (256, 256, 1)

Conv1 = Conv2D(
    filters=64, kernel_size=(3, 3),
    padding='same', data_format='channels_last',
    kernel_initializer=sobel_x)(Pool)
Activ1 = Activation('sigmoid')(Conv1)
BN1 = BatchNormalization(axis=2)(Activ1)
Drop1 = Dropout(0.1)(BN1)

Conv2 = Conv2D(
    filters=64, kernel_size=(5, 5),
    padding='same', data_format='channels_last')(Drop1)
Activ2 = Activation('sigmoid')(Conv2)
BN2 = BatchNormalization(axis=2)(Activ2)
Drop2 = Dropout(0.1)(BN2)

Conv3 = Conv2D(
    filters=64, (7, 7), padding='same',
    data_format='channels_last')(Drop2)
Activ3 = Activation('sigmoid')(Conv3)
BN3 = BatchNormalization(axis=2)(Activ3)
Drop3 = Dropout(0.1)(BN3)

Conv4 = Conv2D(
    filters=64, (9, 9), padding='same',
    data_format='channels_last')(Drop3)
Activ4 = Activation('sigmoid')(Conv4)
BN4 = BatchNormalization(axis=2)(Activ4)
Drop4 = Dropout(0.1)(BN4)

Flatten4 = Flatten()(Drop4)

Dense_XYs = Dense(30*4)(Flatten4)

Activ_XYs = Activation('tanh')(Dense_XYs)

Lambda_XYs = Lambda(lambda x: K.clip(256*(x + 1), 0, 256))(Activ_XYs)

Out_XYs = Reshape((30, 4), name='XYs_Out')(Lambda_XYs)

model = keras.Model(inputs=model_input, outputs=[Out_XYs])

# Define optimizer
optimizer = keras.optimizers.Adadelta()

# Define losses
losses = {
    'XYs_Out': 'mean_squared_error'}

# Define loss weights
weights = {
    'XYs_Out': 1}

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=weights,
    metrics=['mean_squared_error']
    )

if (__name__ == '__main__'):
#    assert len(sys.argv) == 3, 'Pass me both the training and save filepaths!'
    # XXX Testing constants - Remove
    try:
        TRAINING_SET = sys.argv[1]
        SAVE_PATH = sys.argv[2]
    except IndexError:
        print('Pass me both the training set and save filepaths!')
        TRAINING_SET = '../data/train_set_lines.pkl' # HINT input('What\'s the training set filepath?')
        TESTING_SET = '../data/test_set_lines.pkl'
        SAVE_PATH = '../models/saved_model_lines.h5' # HINT input('What\'s the saved model filepath?')
#        sys.exit()

    # Load the training set from the pickled ImageBundle
    train_bundle = pickle.load(open(TRAINING_SET, 'rb'))
    train_X = train_bundle.images
    train_y = train_bundle.line_list

    # Load the testing set from the pickled ImageBundle
    test_bundle = pickle.load(open(TESTING_SET, 'rb'))
    test_X = test_bundle.images
    test_y = test_bundle.line_list

    # Output matching
    training_outs = {
        'XYs_Out': train_y[:, :, :, 0]}

    # Fit the model to the training ImageBundle
    model.fit(
        train_X,
        training_outs,
        epochs=5,
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
    testing_outs = {
        'XYs_Out': test_y[:, :, :, 0]}

    print(model.evaluate(
            test_X,
            testing_outs))
