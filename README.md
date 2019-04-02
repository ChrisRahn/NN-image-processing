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

-   IN: 512x512 px greyscale image (1 channel)
-   1. 2D Convolution, 3x3 kernel, stride 1, 15 filters
-   2. MaxPool, size 2
-   3. 2D Convolution, 3x3 kernel, stride 1, 15 filters
-   4. MaxPool, size 2
-   5. Flatten operation
-   6. Dense layer, 300 nodes
-   7. Dense layer, 150 nodes
-   8. Parametric ReLU activation layer
-   9. Reshape operation to 30x5 output

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
- Re-do loss function to invoke the output drawing methods, then compare result to input image
- Test different conv. kernel initializers (Sobel?)
- Handling for differently sized images (currently 512x512 px only)
- Sort out RGB channel handling
- Enforce more consistent filename I/O
- TensorFlow logging, early stopping?
- Expand model to more shape channels (currently triangles only)
- Web app framework to ingest a new image and report result
- Better Markdown in this README!