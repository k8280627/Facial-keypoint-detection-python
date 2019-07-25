## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input

        # input image = 224 x 224
        self.conv1 = nn.Conv2d(1, 32, 5) #(224 - 5)/1 + 1 = 220 output, will be 110 x 110 feature map after pooling
        self.conv2 = nn.Conv2d(32, 64, 5) #(110 - 5)/ 1 + 1 = 106 output size, will be 54 x 54 feature map after pooling
        self.conv3 = nn.Conv2d(64, 128, 5) #(53 - 5)/1 + 1 = 49 output size, will be 26 x 26 feature map after pooling
        self.conv4 = nn.Conv2d(128, 256, 5) #(24 - 5)/1 +1 = 20, 12 x 12

        self.pool = nn.MaxPool2d(2, 2)   
        self.fc1 = nn.Linear(256 * 10 * 10, 1024) #(W-F)/S +1 = output of conv layer
        
        self.fc1_drop = nn.Dropout(p=0.5)
        self.conv_drop = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(1024, 136)

        
    def forward(self, x):
        ## x is the input image
        x = self.conv_drop(self.pool(F.relu(self.conv1(x))))
        x = self.conv_drop(self.pool(F.relu(self.conv2(x))))
        x = self.conv_drop(self.pool(F.relu(self.conv3(x))))
        x = self.conv_drop(self.pool(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1) #flatten the x.
    
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        return x
