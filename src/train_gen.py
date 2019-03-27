#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a training set for the model
"""
import numpy as np
import cairo
# Simple arrays for testing


def simple_tri():
    return np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0.5, 1, 1, 1, 1, 1, 1, 1, 0.5, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def simple_test():
    return np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0.5, 1, 0.5, 0, 0, 0, 0.5, 1, 0.5, 0],
                     [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                     [0.5, 1, 1, 1, 0.5, 0, 0.5, 1, 1, 1, 0.5],
                     [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]])

    
class drawing_bundle():
    def __init__(self):
        # Create empty shapelists like (index, pos_x, pos_y, w_scale, h_scale, )
        self.shapelist=np.zeros(())


if (__name__ == '__main__'):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context()
    # TODO Drawing functions...
    buf = surface.get_data()
    data = np.ndarray(shape=(width, height),
                     dtype=np.uint32,
                     buffer=buf)