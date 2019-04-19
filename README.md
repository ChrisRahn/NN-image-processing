# Fracture: Computer Vision to Break It Down

Version 0.4.0

Chris Rahn, April 2019

### DEPENDENCIES

- Tensorflow 1.13.1
- Keras 2.2.4-tf
- NumPy 1.15.4
- Cairo 1.14.12
- PIL 1.1.7
- Flask 1.0.2

## INTRO

Computer vision is a subfield of machine learning that applies neural network models to extract information from digital images. Often, these models must make various assumptions about the input images, such as the camera's orientation or the scale of relevant features. Fracture seeks to develop a universally-applicable computer vision model capable of extracting the most basic geometric features from any image to serve as a basis for further specification.

----
## METHOD

Fracture implements a convolutional neural network (CNN) to process any JPEG or PNG image passed by the user and display a line segment representing the most dominant geometric feature captured in the image. This roughly captures both the subject's size and orientation relative to the frame. The line's coordinates are also output, allowing a downstream application to automatically crop, rotate, or resize it, for example.

The source code implements custom classes and methods in order to handle data I/O, as well as create data sets of randomly-drawn shapes to serve as training data. These scripts use the image processing libraries PyCairo and pillow (PIL). The neural network itself is built and fit in TensorFlow via Keras, in the form detailed below - in its current version, the model was trained on a set of 35,000 images. The fitted model is saved for implementation in a custom web app written in Flask, which allow the user to choose image files from their hard drive and see the model outputs in real time.

----
## NEURAL NETWORK

The heart of the model consists of a Tensorflow neural network model implementing the following operations, in order:

INPUT: A JPEG or PNG file of any size   
1. Input array reshaped to represent a 50x50 px greyscale image   
2. Normalize brightness values   
3. 2D convolution using a 3x3 outline kernel    
4. Flatten operation   
5. Densely-connected layer, 2500 nodes   
6. Densely-connected output layer, 4 nodes   
7. Reshape to 1x4 prediction of (x1, y1, x2, y2)   

## RESULTS

Computer vision models are notoriously difficult to train - Fracture is no exception. Many different model configurations and training regimes were attempted, however none were able to outperform a mean absolute error of about 25%. This means that each of the four coordinates defining the model's outputs has an expected error of 25% of the image canvas' corresponding dimension. While this often produces poor estimations that don't align with visual inspection, Fracture shows promising initial results, especially on simple, high-contrast images of singular objects. The model is more successful at capturing the subject's extents and orientation for such images.

![](examples/screenshot.png?raw=true)

## FUTURE WORK

While the current version of Fracture only roughly fits a single line to each image, future versions would work to better represent the complexity of input images, by increasing both the number and type of shapes used.
