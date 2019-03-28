#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator script for training image-bundles
"""
import numpy as np
import math
import cairo
import pickle


class ImageBundle():
    def __init__(self, batch_size, num_tri, width, height):
        # Bundle individual images (array)
        self.images = np.empty((batch_size, height, width, 1))

        # Bundle the arrays of triangle info as a separate attribute
        self.tri_list = np.empty((batch_size, num_tri, 5, 1))
        for i in range(batch_size):
            new_img = CustomImage(width, height)
            new_img.draw_tri(num_tri)
            self.images[i, :, :, 0] = new_img.img[:, :, 0]  # !!!Red channel
            self.tri_list[i, :, :, 0] = new_img.triangles[:, :]

    def save(self, filepath):
        pickle.dump(self, open(filepath, 'wb'))


class CustomImage():
    def __init__(self, width, height):
        self.WIDTH, self.HEIGHT = width, height
        # Create empty shapelist
        # Each row repr one shape with [pos_x, pos_y, w_scale, h_scale, rot]
        self.triangles = None

        # ??? Store a separate array of shape types?
        # self.shape_types = ...

        # Initialize an array of white RGBA values (4 channel)
        self.img = np.zeros((self.HEIGHT, self.WIDTH, 4), dtype=np.uint8)
        # Initialize a Cairo context and drawing surface
        self.surface = cairo.ImageSurface.create_for_data(
            self.img,
            cairo.FORMAT_ARGB32,
            self.WIDTH, self.HEIGHT)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_source_rgb(1.0, 1.0, 1.0)
        self.ctx.paint()

    def triangle(self):
        '''Create a basic equilateral triangle
           Middle-centered w/ side length 0.25*WIDTH'''
        WIDTH, HEIGHT = self.WIDTH, self.HEIGHT
        ctx = self.ctx
        ctx.move_to(0, -0.144*HEIGHT)
#        ctx.rel_line_to(0.5*math.cos(-math.pi/3),
#                        0.5*math.sin(-math.pi/3))
#        ctx.rel_line_to(-0.5, 0)
        ctx.line_to(0.125*WIDTH, 0.072*HEIGHT)
        ctx.line_to(-0.125*WIDTH, 0.072*HEIGHT)
        ctx.close_path()

    def draw_tri(self, num_tri):
        ''' Draw random triangles using triangle() as a template'''
        WIDTH, HEIGHT = self.WIDTH, self.HEIGHT
        ctx = self.ctx
        self.triangles = np.empty((num_tri, 5))
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        for i in range(num_tri):
            ctx.identity_matrix()  # Reset the drawing transformation

            # Randomize drawing params
            off_x = WIDTH * np.random.rand()
            off_y = HEIGHT * np.random.rand()
            w_scale = np.clip(2 * np.random.rand(), 0.1, 4)
            h_scale = np.clip(2 * np.random.rand(), 0.1, 4)
            rot = 2 * math.pi * np.random.rand()

            # Set drawing transformation, then stamp triangle template
            ctx.translate(off_x, off_y)
            ctx.rotate(rot)
            ctx.scale(w_scale, h_scale)
            self.triangle()
            ctx.fill()
            self.triangles[i, :] = [off_x, off_y, w_scale, h_scale, rot]


if (__name__ == '__main__'):
#    test = CustomImage(512, 512)
#    test.draw_tri(5)
#    print(test.img[:, :, 0])
#    test.surface.write_to_png('test.png')
#    print(test.triangles)
    my_bundle = ImageBundle(5, 5, 512, 512)
    my_bundle.save('../data/train_set_01.pkl')
