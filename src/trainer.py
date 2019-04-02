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

# Define simple Sobel kernels
# XXX kernel_tri = tf.constant_initializer(kernels.triangle_5())
# XXX kernel_const = min_max_norm(0.001, None, rate=1, axis=0)
# XXX kernel_nonneg = non_neg()
kernel_sobel_x = tf.constant_initializer(kernels.sobel_x())
kernel_sobel_y = tf.constant_initializer(kernels.sobel_y())

sess = tf.Session()


def build_pred_img(y_pred):
    y_pred_plhdr = tf.placeholder(tf.float32, shape=(512, 512, 1))
    y_pred = y_pred_plhdr
    shape_arr = y_pred.eval(
        feed_dict={y_pred_plhdr: np.random.rand(512, 512, 1)},
        session=sess)
    y_pred_img = OutputImage(512, 512, shape_arr).img[:, :, 0]
    return K.constant(y_pred_img)

# TensorFlow expects 4D tensors of shape (samples, rows, cols, channels)
# Note that the first index (the sample index out of the batch) is stripped
model = keras.Sequential([

        # Convolve the pooled image by the shape kernel(s)
        # ??? Use LocallyConnected2D instead?
        keras.layers.Conv2D(
            input_shape=(512, 512, 1),
            filters=15,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            activation='sigmoid',
            use_bias=True),
#            kernel_initializer=kernel_sobel_x),
            # kernel_constraint=kernel_nonneg),
#        keras.layers.Conv2D(
#            filters=1,
#            kernel_size=(3, 3),
#            strides=(1, 1),
#            padding='same',
#            data_format='channels_last',
#            activation='sigmoid',
#            use_bias=True,
#            kernel_initializer=kernel_sobel_y),

        # Maxpool the image
        keras.layers.MaxPool2D(
            # input_shape=(512, 512, 1),
            pool_size=2,
            padding='same',
            data_format='channels_last'),

        keras.layers.Conv2D(
            filters=15,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            activation='sigmoid',
            use_bias=True),

        # Maxpool the image
        keras.layers.MaxPool2D(
            # input_shape=(512, 512, 1),
            pool_size=2,
            padding='same',
            data_format='channels_last'),

        # Flatten
        keras.layers.Flatten(),

        # Basic Dense layer
        keras.layers.Dense(
            units=300,
            activation=None,
            # kernel_constraint=kernel_nonneg,
            use_bias=True),

        # Basic Dense layer
        keras.layers.Dense(
            units=150,  # !!! 30 output shapes per channel
            activation='relu'),

        # Activation layer
        keras.layers.PReLU(),

        # Reshape & output
        keras.layers.Reshape((30, 5)),

        keras.layers.Lambda(build_pred_img)
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
    w = math_ops.cast(np.array([[1/512, 1/512, 1/4, 1/4, 1/(2*np.pi)]]), y_pred.dtype)
    w = K.transpose(w)
    return K.mean(math_ops.square(K.dot((y_true - y_pred),w)), axis=-1)

def img_to_img(y_true_imgs, y_pred_batch):
    '''Unpack a (5, 30, 5) tensor of batch model outputs
    Create an OutputImage for each
    Compare to the model input, which is already an image'''
#    shape_arr_lst = K.batch_get_value([y_pred_batch])
#    shape_arr_imgs = np.empty(shape=(5, 512, 512))
#    for i, shape_arr in enumerate(shape_arr_lst):
#        shape_arr_imgs[i, :, :] = OutputImage(512, 512, shape_arr).img[:, :, 0]
##    y_pred_imgs = tf.Variable(shape_arr_imgs)
#    y_pred_imgs = tf.constant(0)
#    return K.mean(math_ops.square(y_true_imgs - y_pred_imgs))
    return K.mean(math_ops.square(y_true_imgs))
    
## XXX Add the custom loss to the loss dictionary
#tf.losses.add_loss(scaled_mse)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['mean_squared_error'])


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
    train_y = train_bundle.images

    # Load the testing set from the pickled ImageBundle
    test_bundle = pickle.load(open(TESTING_SET, 'rb'))
    test_X = test_bundle.images
    test_y = test_bundle.images

    # IN: (samples, rows, cols, channels)
    IN_SHAPE = train_X.shape
    # OUT: (samples, shape_idx, shape_attrs, channels)
    OUT_SHAPE = train_y.shape
    # Initialize the training set

    # Fit the model to the training ImageBundle
    model.fit(
        train_X,
        train_y,
        epochs=10,
        verbose=1,
        batch_size=5)

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
