Fracture - An Image Decomposition Engine (In Progress)

Version 0.1.0

Chris Rahn

March 2019

----------
PURPOSE
-----------
To de-construct a given image into its simplest geometric elements for natural feature recognition at-scale. This is acheived with a convolutional neural network that processes the image at various feature sizes.

----------
MODEL
----------
(in flux)

A sequential Tensorflow neural network with these layers:

-    1. MaxPool with size 2 window (IN: 512x512 px, 1 channel = 512x512x1, OUT: 256x256x1)
-    2. 2D Convolution with 8x8 kernel, stride 8, 5 filters (OUT: 64x64x5)
-    3. 2D Convolution with 8x8 kernel, stride 8, 5 filters (OUT: 8x8x25)
-    4. Flatten operation (OUT: 1600)
-    5. Densely-connected layer, 25 nodes (OUT: 25)
-    6. PReLU (Parametric ReLU) activation function (OUT: 25)
-    7. Reshaping to 5x5 output (OUT: 5x5)

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
- Handling for different amounts of shapes (currently 5 only)
- Test different loss functions (Jaccard Sim?) to compare atts. fairly
- Test different conv. kernel initializers (Sobel?)
- Handling for differently sized images (currently 512x512 px only)
- Sort out RGB channel handling
- TensorFlow logging, early stopping?
- Expand model to more shape channels (currently triangles only)
- More NN layers? (Try another Dense)
- Web app framework to ingest a new image and report result
- Better Markdown in this README!