## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # image shape is 224 * 224
        
        # convolution layer with (5,5) kernal size and number of kernals is 32
        # after applying convolution, output shape = (220, 220)
        # output size after apply 5 * 5 max pooling = (32, 44, 44)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(5)
        
        # convolution layer with (4,4) kernal size and number of kernals is 64
        # after applying convolution, output shape = (220, 220)
        # output size after apply 4 * 4 max pooling = (64, 55, 55)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.pool2 = nn.MaxPool2d(4)
        
        # convolution layer with (3,3) kernal size and number of kernals is 128
        # after applying convolution, output shape = (54, 54)
        # output size after apply 3 * 3 max pooling = (128, 18, 18)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(3)
        
        
        # convolution layer with (2,2) kernal size and number of kernals is 256
        # after applying convolution, output shape = (18, 18)
        # output size after apply 5 * 5 max pooling = (256, 9, 9)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.pool4 = nn.MaxPool2d(1)
        
        
        self.dense1 = nn.Linear(256, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 136)
        self.drop = nn.Dropout(p=0.4)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # flatten layer
        x = x.view(x.size(0), -1)
        
        
        x = self.dense1(x)
        x = F.relu(x)
        x = self.drop(x)
        
        x = self.dense2(x)
        x = F.relu(x)
        
        x = self.drop(x)
        
        x = self.dense3(x)
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
