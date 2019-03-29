#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the input image through the neural network model
"""
import tensorflow as tf

saver = tf.train.Saver()

if (__name__ == '__main__'):
    with tf.Session() as sess:
        saver.restore(sess, '../checkpoints/fracture.ckpt')
        print('Successfully restored model')
