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

**Total Parameters for LeNet-5:**  
156 (conv1) + 2416 (conv2) + 3080 (fc1) + 10164 (fc2) + 850 (fc3) = 17166 parameters  

## Number of Parameters for the CustomMLP model
For the CustomMLP model, there are two fully connected layers.  
We calculate the number of parameters for the fully connected layers with this formula: (input_features x output_features) + bias  

First fully connected layer: Input features = 28 x 28 (size of the input image), Output features = 20  
Number of parameters = (28 x 28 x 20) + 20 = 15620    

Second fully connected layer: Input features = 20, Output features = 10 (for 10 classes)  
Number of parameters = (20 x 10) + 10 = 210    

**Total Parameters for CustomMLP:**  
15620 (fc1) + 210 (fc2) = 15830 parameters  

## Models' Performance
Two models' performance (without applying any regularization techniques) can be seen in the following graphs. Graphs show the training and test accuracy and loss values through 5 epochs.
Final values for loss and accuracy for the LeNet-5 model:  
Training Loss=0.0301, Accuracy=99.06%  
Test Loss=0.0363, Accuracy=98.78%  
Final values for loss and accuracy for the CustomMLP model:  
Training Loss=0.1362, Accuracy=96.00%  
Test Loss=0.1575, Accuracy=95.27%  
<img width="1014" alt="Ekran Resmi 2024-04-16 ÖS 5 01 22" src="https://github.com/beyzakebeli/MNIST_classification/assets/92715108/0f236728-7246-4ee1-b6b6-0a20e5439ffc">
As expected, the LeNet-5 model's performance is better than the CustomMLP model since the CustomMLP's structure is so much simpler than LeNet-5. Convolutional layers are well-suited for extracting hierarchical features from spatial data and pooling layers help extracting the dominant features. Existence of these layers in the LeNet-5 structure makes the performance of the model better. Also, number of parameters for the models are similar but still LeNet-5 has more parameters than CustomMLP which makes the learning capacity of the model higher.

## Regularization Methods
I used two regularization methods: **L2 Regularization** and **Dropout Method** to improve the performance of the LeNet-5 model further.  
Dropout rate is set to 0.5 and weight decay value is set to 0.0001 in the train() function.  
Performance of the LeNet-5 model after regularization:  
Training Loss=0.0358, Accuracy=99.08%  
Test Loss=0.0309, Accuracy=99.01%  

