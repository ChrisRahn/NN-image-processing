#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator script for training image-bundles
"""
import numpy as np
import io
import math
import cairo
import pickle
from PIL import Image


class ImageBundle():
    '''A collection of randomly-generated CustomImages'''
    def __init__(self, batch_size, width, height, num_tri=0, num_lines=0, num_points=0):
        # Bundle individual images (array)
        self.images = np.empty((batch_size, height, width, 1))

        # Bundle the arrays of triangle info as a separate attribute
        self.tri_list = np.empty((batch_size, num_tri, 5, 1))
        self.line_list = np.empty((batch_size, num_lines, 4, 1))
        self.point_list = np.empty((batch_size, 2*num_lines, 2, 1))

        for i in range(batch_size):
            new_img = CustomImage(width, height)
            if num_tri:
                new_img.rand_tri(num_tri)
                self.tri_list[i, :, :, 0] = new_img.triangles[:, :]
                # Reorder tri_list from largest to smallest (by w_scale*h_scale)
                areas = self.tri_list[i, :, 2, 0] * self.tri_list[i, :, 3, 0]
                sorted_areas = self.tri_list[i, np.argsort(areas)[-1::-1], :, 0]
                self.tri_list[i, :, :, 0] = sorted_areas

            if num_lines:
                new_img.rand_line(num_lines)
                self.line_list[i, :, :, 0] = new_img.lines[:, :]
                self.point_list[i, :, :, 0] = new_img.lines[:, :].reshape(2*num_lines, 2)

#            if num_points:
#                new_img.rand_point(num_points)
#                self.point_list[i, :, :, 0] = new_img.points[:, :]

            self.images[i, :, :, 0] = new_img.img[:, :, 0]  # !!!Red channel

    def save(self, filepath):
        pickle.dump(self, open(filepath, 'wb'))


class CustomImage():
    '''A randomly-generated image object used to train the NN model
    IN: image width=256px, image height=256px'''
    def __init__(self, width=256, height=256):
        self.WIDTH, self.HEIGHT = width, height

        # Array of triangles will be like [pos_x, pos_y, w_scale, h_scale, rot]
        self.triangles = None

        # Array of line segments will be like [x1, y1, x2, y2]
        self.lines = None

        # Array of points like [x1, y1]
        self.points = None

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

    def draw_tri(self, off_x, off_y, w_scale, h_scale, rot, alpha=0.5):
        ''' Draw a particular triangle using triangle() as a template'''
        ctx = self.ctx
        # Reset drawing transformation and choose black source
        ctx.identity_matrix()
        ctx.set_source_rgba(0.0, 0.0, 0.0, alpha)
        # Set drawing transformation, then stamp triangle template
        ctx.translate(off_x, off_y)
        ctx.rotate(rot)
        ctx.scale(w_scale, h_scale)
        self.triangle()
        ctx.fill()

    def rand_tri(self, num_tri):
        '''Draw a particular number of random triangles'''
        WIDTH, HEIGHT = self.WIDTH, self.HEIGHT
        self.triangles = np.empty((num_tri, 5))

        for i in range(num_tri):
            # Randomize drawing params
            off_x = WIDTH * np.random.rand()
            off_y = HEIGHT * np.random.rand()
            w_scale = np.clip(2 * np.random.rand(), 0.1, 4)
            h_scale = np.clip(2 * np.random.rand(), 0.1, 4)
            rot = 2 * math.pi * np.random.rand()

            self.draw_tri(off_x, off_y, w_scale, h_scale, rot, alpha=0.5)
            self.triangles[i, :] = [off_x, off_y, w_scale, h_scale, rot]

    def draw_line(self, x1, y1, x2, y2, alpha=1):
        WIDTH, HEIGHT = self.WIDTH, self.HEIGHT
        ctx = self.ctx
        ctx.identity_matrix()  # Reset drawing transformation
        ctx.set_source_rgba(0.0, 0.0, 0.0, alpha)  # Black source
        ctx.set_line_width(6.0)  # Line width
        ctx.move_to(x1*WIDTH, y1*HEIGHT)
        ctx.line_to(x2*WIDTH, y2*HEIGHT)
        ctx.stroke()

    def rand_line(self, num_lines):
        self.lines = np.empty((num_lines, 4))

        for i in range(num_lines):
            # Randomize drawing params
            x1, x2 = np.random.rand(2)
            y1 = np.random.rand() / 2  # Upper half of frame
            y2 = 0.5 + np.random.rand() / 2  # Lower half of frame

            self.draw_line(x1, y1, x2, y2, alpha=1)
            self.lines[i, :] = [x1, y1, x2, y2]

    def draw_point(self, x1, y1, alpha=1.0):
        WIDTH, HEIGHT = self.WIDTH, self.HEIGHT
        ctx = self.ctx
        ctx.identity_matrix()
        ctx.set_source_rgba(0.0, 0.0, 0.0, alpha)
        ctx.rectangle(x1*WIDTH, y1*HEIGHT, 10, 10)
        ctx.fill()

    def rand_point(self, num_points):
        self.points = np.empty((num_points), 2)

        for i in range(num_points):
            # Randomize the drawing params
            x1 = np.random.rand()
            y1 = np.random.rand()

        self.draw_point(x1, y1, alpha=1.0)
        self.points[i, :] = [x1, y1]

    def display(self):
        ''' Use PIL to show the image'''
        with Image.fromarray(self.img, mode='RGBA') as out:
            out.show()


class InputImage(CustomImage):
    '''A object just for handling model inputs
    (__init__() and display() overwritten)
    IN: A path to a 256x256 JPEG file'''

    def __init__(self, image_path):
        self.img_in = Image.open(image_path)

        # Add alpha channel if not present
        if 'A' not in self.img_in.getbands():
            self.img_in.putalpha(256)

        # Convert to greyscale mode (for now)
        img_grey = self.img_in.convert('L')

        # Cast into a NumPy array through a Cairo surface
        barr = bytearray(img_grey.tobytes('raw', 'L'))
        self.surface = cairo.ImageSurface.create_for_data(
            barr,
            cairo.FORMAT_A8,
            256, 256)
        buff = self.surface.get_data()
        self.data = np.ndarray(
            shape=(256, 256, 1),
            dtype=np.uint8,
            buffer=buff)

    def display(self):
        self.img_in.show()


class OutputImage(CustomImage):
    '''A subclass of CustomImage just for displaying model outputs
    IN: image width, image height, ?x5 NumPy array of triangle data'''
    def __init__(self, width, height, triangles=None, lines=None, points=None):
        super().__init__(width, height)
        self.triangles = triangles
        self.lines = lines
        self.points = points
        self.update()

    def update(self):
        '''Update the image array with the shapelists'''
        # Reset and clear the drawing surface
        self.ctx.identity_matrix()
        self.ctx.set_source_rgb(1.0, 1.0, 1.0)
        self.ctx.paint()

        # Draw each triangle stored in the triangle array
        if self.triangles is not None:
            for triangle in self.triangles:
                off_x, off_y, w_scale, h_scale, rot = triangle
                self.draw_tri(off_x, off_y, w_scale, h_scale, rot, alpha=0.5)

        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line
                self.draw_line(x1, x2, y1, y2, alpha=1)

        if self.points is not None:
            for point in self.points:
                x1, y1 = point
                self.draw_point(x1, y1, alpha=1.0)


if (__name__ == '__main__'):
    create_bundle_size = int(input('Create how many images?'))
    create_num_tri = int(input('How many triangles per image?'))
    create_num_lines = int(input('How many lines per image?'))
    create_save_path = input('Path (.pkl) to save to?')
    new_bundle = ImageBundle(create_bundle_size, 256, 256, num_tri=create_num_tri, num_lines=create_num_lines)
    new_bundle.save(create_save_path)
    print('Here\'s the first of the new images I just created.')
    OutputImage(256, 256,
        triangles=new_bundle.tri_list[0, :, :, 0],
        lines=new_bundle.line_list[0, :, :, 0]).display()
