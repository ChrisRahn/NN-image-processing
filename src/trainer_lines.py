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

model_input = Input(shape=(512, 512, 1))

Lambda_In = Lambda(lambda x: x/255.)(model_input)
#Pool1 = MaxPool2D(pool_size=(2, 2))(model_input)  # (256, 256, 1)
Pool2 = MaxPool2D(pool_size=(8, 8))(Lambda_In)  # (164, 64, 1)
#Pool3 = MaxPool2D(pool_size=(2, 2))(Pool2)  # (64, 64, 1)
#Pool4 = MaxPool2D(pool_size=(2, 2))(Pool3)  # (32, 32, 1)
#Pool5 = MaxPool2D(pool_size=(2, 2))(Pool4)  # (16, 16, 1)

Conv21 = Conv2D(128, (3, 3), padding='same', kernel_initializer=sobel_x, data_format='channels_last')(Pool2)  # (128, 128, 30)
Activ21 = Activation('sigmoid')(Conv21)
BN21 = BatchNormalization(axis=2)(Activ21)
Drop21 = Dropout(0.1)(BN21)

Conv22 = Conv2D(128, (5, 5), padding='same', data_format='channels_last')(Drop21)
Activ22 = Activation('sigmoid')(Conv22)
BN22 = BatchNormalization(axis=2)(Activ22)
Drop22 = Dropout(0.1)(BN22)

Conv23 = Conv2D(128, (7, 7), padding='same', data_format='channels_last')(Drop22)
Activ23 = Activation('sigmoid')(Conv23)
BN23 = BatchNormalization(axis=2)(Activ23)
Drop23 = Dropout(0.1)(BN23)

Conv24 = Conv2D(128, (9, 9), padding='same', data_format='channels_last')(Drop23)
Activ24 = Activation('sigmoid')(Conv24)
BN24 = BatchNormalization(axis=2)(Activ24)
Drop24 = Dropout(0.1)(BN24)

Flatten24 = Flatten()(Drop24)

Dense2_XYs = Dense(30*4)(Flatten24)

Activ_XYs = Activation('tanh')(Dense2_XYs)

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
