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
        # input image = 224 x 224
        self.conv1 = nn.Conv2d(1, 32, 5) #(224 - 5)/1 + 1 = 220 output, will be 110 x 110 feature map after pooling
        self.conv2 = nn.Conv2d(32, 64, 5) #(110 - 5)/ 1 + 1 = 106 output size, will be 54 x 54 feature map after pooling
        self.conv3 = nn.Conv2d(64, 128, 5) #(53 - 5)/1 + 1 = 49 output size, will be 26 x 26 feature map after pooling
        self.conv4 = nn.Conv2d(128, 256, 5) #(24 - 5)/1 +1 = 20, 12 x 12
        #self.conv5 = nn.Conv2d(256, 512, 1) #(12 - 1) + 1 = 12,   5 x 5 
        self.pool = nn.MaxPool2d(2, 2)   
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.fc1 = nn.Linear(256 * 10 * 10, 1024) #(W-F)/S +1 = output of conv layer
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.5)
        #self.conv1_drop
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(1024, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
    
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
