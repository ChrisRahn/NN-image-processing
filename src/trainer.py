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

#img = imread('../data/triforce.jpg')  # Grey-scale only for now
img = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# Rescale img's values to be [0 - 255]
# img = img * 255.  #TODO Remove prob

# Set training data to be random and test to be the image
train_X = np.random.random(size=img.shape)
train_y = img
# TensorFlow expects 5D tensors of kind (samples, time, rows, cols, channels)
train_X = np.resize(train_X, (1, img.shape[0], img.shape[1], 1))
train_y = np.resize(train_y, (1, img.shape[0], img.shape[1], 1))

# Define a simple triangular kernel
tiny_tri = np.array([[0, 1, 0],
                     [0, 0, 0],
                     [1, 0, 1]])

triangle = np.array([[0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]])

triangle_2 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

kernel_t = tf.constant_initializer(tiny_tri)


def my_rescaler(X):
    max_val = K.max(K.max(X, axis=2, keepdims=True), axis=3)[0, 0, 0, 0]
    min_val = K.min(K.min(X, axis=2, keepdims=True), axis=3)[0, 0, 0, 0]
    return 255*(X - min_val)/(max_val - min_val)

#%% #!!! DELET THIS
## Define a simple LSTM model
#model = keras.Sequential([
#        keras.layers.ConvLSTM2D(
#             input_shape=(None, train_X.shape[2], train_X.shape[3], 1),
#             filters=11,
#             kernel_size=(3, 3),
#             padding='same',
##                 strides=1,
##                 dilation_rate=(10, 10),
#             activation='relu',
##             recurrent_activation='relu',
##                 use_bias=True,
#             kernel_initializer=kernel_t,
##                 recurrent_initializer=None,
##                 kernel_constraint=None,
#             return_sequences=True,
##                 return_state=False,
##                 stateful=True,
#             data_format='channels_last'),
#        keras.layers.Lambda(lambda X: my_rescaler(X))
#
##        keras.layers.BatchNormalization()
##HINT   keras.layer.Permute(...)  # To apply the different kernels?
#        ])
#%%
    
model = keras.Sequential([
        keras.layers.AveragePooling2D(
                input_shape=(None, train_X.shape[1], train_X.shape[2], 1),
                pool_size=(2, 2),
                padding='same',
                data_format='channels_last')
        ])

# Compile the model
model.compile(optimizer='adadelta',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Fit the model to the training image
model.fit(train_X, train_y, epochs=5, verbose=2, batch_size=1)

# Evaluate model
# TODO model.evaluate(...)

# Predict a random array for testing
test_X = np.resize(np.random.random(size=img.shape), (1, 1, img.shape[0], img.shape[1], 1))
pred_y = model.predict(test_X)
pred_y_img = pred_y[0, 0, :, :, :]
imshow(pred_y_img[:, :, 0])
print('Mean values are: ', pred_y_img.mean(axis=0).mean(axis=0))
