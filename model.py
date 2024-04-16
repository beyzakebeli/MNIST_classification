import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # 2 convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 

        # 3 fully connected layers and the last one gives the output
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output size=10 (for 10 classes)

        # ReLU activation function
        self.activation = nn.ReLU()

        # 2 pooling layers with max-pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout method for regularization
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, img):

        output = self.activation(self.conv1(img))
        output = self.pool1(output)

        output = self.activation(self.conv2(output))
        output = self.pool2(output)

        output = torch.flatten(output, 1)

        output = self.activation(self.fc1(output))
        output = self.activation(self.fc2(output))
        output = self.fc3(output)
        
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 20)       
        self.fc2 = nn.Linear(20, 10)        # Output size: 10 (for 10 classes)

        # Using the same activation function as LeNet5
        self.activation = nn.ReLU()

    def forward(self, img):

        output = torch.flatten(img, 1)
        output = self.activation(self.fc1(output))
        output = self.fc2(output)

        return output
