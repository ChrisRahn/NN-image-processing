#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the neural network model
"""

import pickle
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from kernels import outline

# Define outline filter
outline = tf.constant_initializer(outline())

# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped

# Define the model's layers
model_input = kl.Input(shape=(50, 50, 1))

Lambda_In = kl.Lambda(lambda x: (x-128)/64.)(model_input)

Activ_In = kl.Activation('tanh')(Lambda_In)

Conv1 = kl.Conv2D(
    filters=1, kernel_size=(3, 3),
    padding='same', data_format='channels_last',
    activation='tanh',
    kernel_initializer=outline
    )(Activ_In)

FlattenAll = kl.Flatten()(Conv1)

Dense_Int = kl.Dense(2500, activation='tanh')(FlattenAll)

Dense_XYs = kl.Dense(1*4, activation='sigmoid')(Dense_Int)

Out_XYs = kl.Reshape((1, 1, 4, 1), name='XYs_Out')(Dense_XYs)

# Invoke model from above layers
model = keras.Model(inputs=model_input, outputs=[Out_XYs])

# Define optimizer
optimizer = keras.optimizers.Adadelta()

# Define losses
losses = {
    'XYs_Out': 'mean_squared_error'}

# Define loss weights
weights = {
    'XYs_Out': 1000000000}  # Extra high to show enough sigfigs

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=weights,
    metrics=['mean_squared_error', 'mae'])

if (__name__ == '__main__'):

    try:
        TRAINING_SET = sys.argv[1]
    except IndexError:
        TRAINING_SET = input('What\'s the training set filepath?')

    try:
        TESTING_SET = sys.argv[2]
    except IndexError:
        TESTING_SET = input('What\'s the testing set filepath?')

    try:
        SAVE_PATH = sys.argv[3]
    except IndexError:
        SAVE_PATH = input('Where should I save the model?')

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
        'XYs_Out': train_y}

    print('\nFitting the model...')

    # Suppress TensorFlow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Fit the model to the training ImageBundle
    model.fit(
        train_X,
        training_outs,
        epochs=120,
        verbose=1,
        batch_size=20,
        validation_split=0.1)

    # Write model config to YAML
    model_yaml = model.to_yaml()
    with open('../models/model_config.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)

    # Save model
    model.save(SAVE_PATH, overwrite=True, include_optimizer=True)
    print('\nModel saved at: %s' % SAVE_PATH)

    # Model evalutaion
    testing_outs = {
        'XYs_Out': test_y}

    print(model.evaluate(
            test_X,
            testing_outs))
