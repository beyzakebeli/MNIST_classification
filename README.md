# MNIST_classification

## Number of Parameters for the LeNet-5 model

![LeNet-5 architecture](https://github.com/beyzakebeli/MNIST_classification/assets/92715108/f107594b-a074-45a6-ae87-6eb9d01f5587)

We calculate the number of parameters for the convolutional layers with this formula: (input_channels x output_channels x kernel_height x kernel_width) + bias

For the first convolutional layer: Input channels = 1, Output channels = 6, Kernel size = 5x5
Number of parameters = (1 x 6 x 5 x 5) + 6 = 156

For the second convolutional layer: Input channels = 6, Output channels = 16, Kernel size = 5x5
Number of parameters = (6 x 16 x 5 x 5) + 6 = 2416

We calculate the number of parameters for the fully connected layers with this formula: (input_features x output_features) + bias
We have three fully connected layers.

First fully connected layer: Input features = 16 x 4 x 4 (output from conv2), Output features = 120
Number of parameters = (16 x 4 x 4 x 120) + 120 = 3080

Second fully connected layer: Input features = 120, Output features = 84
Number of parameters = (120 x 84) + 84 = 10164

Third fully connected layer: Input features = 84, Output features = 10 (for 10 classes)
Number of parameters = (84 x 10) + 10 = 850

Total Parameters for LeNet-5:
156 (conv1) + 2416 (conv2) + 3080 (fc1) + 10164 (fc2) + 850 (fc3) = 17166 parameters
