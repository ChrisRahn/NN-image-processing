#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a training set for the model
"""
import numpy as np
import math
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


class ImageBundle():
    def __init__(self):
        # Individual images are array objects
        # Bundle images together as (img_idx, num_shapes, shape_attr)
        self.images = np.empty_like((5, 3, 5))


class CustomImage():
    def __init__(self):
        self.WIDTH, self.HEIGHT = 512, 512
        # Create empty shapelist
        # Each row repr one shape with [pos_x, pos_y, w_scale, h_scale, rot]
        self.triangles = np.zeros((512, 512))
        # ??? Store a separate array of shape types
        # self.shape_types =
        # Initialize a Cairo context and drawing surface
        self.img = np.ndarray()
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 512, 512)
        self.ctx = cairo.Context(surface)
        self.ctx.set_source_rgb(1, 1, 1)
        self.ctx.paint()
        self.ctx.set_source_rgb(0, 0, 0)

    def triangle(self):
        # Create just the path for a basic equilateral triangle
        self.ctx.scale(self.WIDTH, self.HEIGHT)
        self.ctx.move_to(0.50*WIDTH, 0.17*HEIGHT)
        self.ctx.line_to(0.75*WIDTH, 0.625*HEIGHT)
        self.ctx.line_to(0.25*WIDTH, 0.625*HEIGHT)
        self.ctx.close_path()
        self.ctx.fill()

    def create_image(self):
        # Create a standard triangle on the drawing surface



if (__name__ == '__main__'):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context()
    # TODO Drawing functions...
    buf = surface.get_data()
    data = np.ndarray(shape=(width, height),
                     dtype=np.uint32,
                     buffer=buf)