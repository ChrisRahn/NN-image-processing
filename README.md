Fracture - An Image Decomposition Engine (In Progress)

Chris Rahn
March 2019

PURPOSE
To de-construct a given image into its simplest geometric elements for natural feature recognition at-scale. This is acheived with a convolutional neural network that processes the image at various feature sizes.

MODEL
(in flux)
A sequential Tensorflow neural network with these layers:
    1. MaxPooling by current feature size (1 channel)
    2. 2D Convolution with current shape kernel (1 channel, 1 filter)
    3. Flatten operation
    4. Densely-connected layer (81 nodes)
    5. Densely-connected layer (3 nodes)
    6. PReLU (Parametric ReLU) output layer

DEPENDENCIES
Tensorflow
Keras
NumPy
Cairo

FUTURE WORK

