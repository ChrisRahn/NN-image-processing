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
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, BatchNormalization, Dropout, Flatten, Concatenate, Dense, Reshape, Activation, Lambda, LeakyReLU

# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped

model_input = Input(shape=(512, 512, 1))

Lambda_In = Lambda(lambda x: x/255.)(model_input)
#Pool1 = MaxPool2D(pool_size=(2, 2))(model_input)  # (256, 256, 1)
Pool2 = MaxPool2D(pool_size=(8, 8))(Lambda_In)  # (164, 64, 1)
#Pool3 = MaxPool2D(pool_size=(2, 2))(Pool2)  # (64, 64, 1)
#Pool4 = MaxPool2D(pool_size=(2, 2))(Pool3)  # (32, 32, 1)
#Pool5 = MaxPool2D(pool_size=(2, 2))(Pool4)  # (16, 16, 1)

Conv21 = Conv2D(64, (3, 3), padding='same', data_format='channels_last')(Pool2)  # (128, 128, 30)
Activ21 = Activation('sigmoid')(Conv21)
BN21 = BatchNormalization(axis=2)(Activ21)
Drop21 = Dropout(0.1)(BN21)

Conv22 = Conv2D(64, (5, 5), padding='same', data_format='channels_last')(Drop21)
Activ22 = Activation('sigmoid')(Conv22)
BN22 = BatchNormalization(axis=2)(Activ22)
Drop22 = Dropout(0.1)(BN22)

#Conv23 = Conv2D(64, (7, 7), padding='same', data_format='channels_last')(Drop22)
#Activ23 = Activation('sigmoid')(Conv23)
#BN23 = BatchNormalization(axis=2)(Activ23)
#Drop23 = Dropout(0.1)(BN23)

Flatten22 = Flatten()(Drop22)
#Conc = Concatenate(axis=-1)([Flatten21, Flatten22])

Dense2_Pos = Dense(30*2)(Flatten22)
Dense2_Siz = Dense(30*2)(Flatten22)
Dense2_Rot = Dense(30*1)(Flatten22)

Activ_Pos = Activation('tanh')(Dense2_Pos)
Activ_Siz = Activation('tanh')(Dense2_Siz)
Activ_Rot = Activation('tanh')(Dense2_Rot)

Lambda_Pos = Lambda(lambda x: 256*(x + 1))(Activ_Pos)
Lambda_Siz = Lambda(lambda x: 2*(x + 1))(Activ_Siz)
Lambda_Rot = Lambda(lambda x: 3.14*(x + 1))(Activ_Rot)

Out_Pos = Reshape((30, 2), name='Position')(Lambda_Pos)
Out_Siz = Reshape((30, 2), name='Size')(Lambda_Siz)
Out_Rot = Reshape((30,), name='Rotation')(Lambda_Rot)

model = keras.Model(inputs=model_input, outputs=[Out_Pos, Out_Siz, Out_Rot])

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
    'Position': 'mean_squared_error',
    'Size': 'mean_squared_error',
    'Rotation': 'mean_squared_error'}

# Define loss weights
weights = {
    'Position': 1/(512**2),
    'Size': 1/(4**2),
    'Rotation': 1/(6.28**2)}

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
        TRAINING_SET = '../data/train_set_multi.pkl' # HINT input('What\'s the training set filepath?')
        TESTING_SET = '../data/test_set_multi.pkl'
        SAVE_PATH = '../models/saved_model_multi.h5' # HINT input('What\'s the saved model filepath?')
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

    # Output matching
    training_outs = {
        'Position': train_y[:, :, :1, 0],
        'Size': train_y[:, :, 2:4, 0],
        'Rotation': train_y[:, :, 4, 0]}

    # Fit the model to the training ImageBundle
    model.fit(
        train_X,
        training_outs,
        epochs=15,
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
        'Position': test_y[:, :, :1, 0],
        'Size': test_y[:, :, 2:4, 0],
        'Rotation': test_y[:, :, 4, 0]}

    print(model.evaluate(
            test_X,
            testing_outs))
