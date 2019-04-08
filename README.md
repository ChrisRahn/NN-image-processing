Fracture - An Image Decomposition Engine

Version 0.4.0

Chris Rahn

March 2019

----------
PURPOSE
-----------
To de-construct a given image into its simplest geometric elements for natural feature recognition at-scale. This is acheived with a convolutional neural network that processes the image at various feature sizes.

----------
MODEL
----------
A sequential Tensorflow neural network with these layers:

-   IN: JPEG or PNG file of any size
-   1. Input array reshaped to represent a 50x50 px greyscale image
-   2. Normalize brightness values
-   3. 2D convolution using a 3x3 outline kernel
-   4. Flatten operation
-   5. Densely-connected layer, 2500 nodes
-   6. Densely-connected output layer, 4 nodes
-   7. Reshape to 1x4 prediction of (x1, y1, x2, y2)

----------
DEPENDENCIES
----------
- Tensorflow 1.13.1
- Keras 2.2.4-tf
- NumPy 1.15.4
- Cairo 1.14.12
- PIL 1.1.7

----------
TO-DO
----------
- TensorFlow logging & early stopping
- Web app framework to ingest a new image and report result
