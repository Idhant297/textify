import numpy as np
from layers import Conv2DLayer, ReLULayer, MaxPoolingLayer, FullyConnectedLayer

class CNN:
    #basic sturture of the cnn model
    # 2 conv layers
    # 1 max pooling layer
    # 1 conv layer
    # activation layer (relu)
    # flatten the output
    # fully connected layer
    def __init__(self):
        # conv layer 1
        self.conv1 = Conv2DLayer(input_channel = 1, num_filters = 32, kernel_size = 3, stride = 1, padding = 1)
        # conv layer 2
        self.conv2 = Conv2DLayer(input_channel = 32, num_filters = 64, kernel_size = 3, stride = 1, padding = 1)
        # max pooling layer 1
        self.pool1 = MaxPoolingLayer(kernel_size = 2, stride = 2)
        # conv layer 3
        self.conv3 = Conv2DLayer(input_channel = 64, num_filters = 128, kernel_size = 3, stride = 1, padding = 1)
        # activation layer
        self.relu = ReLULayer()
        # fully connected layer
        self.fc1 = FullyConnectedLayer(input_size = self.calculate_fc_input_dim(), num_classes = 26)
    
    
    def calculate_fc_input_dim(self):
        input_size = 28  
        conv1_output_size = (input_size - 3 + 2 * 1) // 1 + 1  # Output size formula: (W−F+2P)/S+1
        conv2_output_size = (conv1_output_size - 3 + 2 * 1) // 1 + 1
        pooled_output_size = (conv2_output_size - 2) // 2 + 1

        conv3_output_size = (pooled_output_size - 3 + 2 * 1) // 1 + 1

        return conv3_output_size * conv3_output_size * 128  # Depth of third conv layer

    def forward(self, input):
        # first two conv layer
        input = self.conv1.forward(input)
        print("After conv1", input.shape)
        input = self.conv2.forward(input)
        print("After conv2", input.shape)   
        
        # max pooling layer
        input = self.pool1.forward(input)
        print("After pool1", input.shape)
        
        # third conv layer
        input = self.conv3.forward(input)
        print("After conv3", input.shape)
        
        # relu activation layer
        input = self.relu.forward(input)
        print("After relu", input.shape)
        
        # flatten the output
        input = input.reshape(input.shape[0], -1)
        print("After flatten", input.shape)
        
        # fully connected layer
        output = self.fc1.forward(input)
        print("output shape", output.shape)
        return output

    def backward(self, dL_dinput):
        # fully connected layer
        dL_dinput = self.fc1.backward(dL_dinput)
        print("After fc1 backward", dL_dinput.shape)
        
        # reshape the output
        #dL_dinput = dL_dinput.reshape(self.conv3.cache[0].shape)
        # or reshape(-1, output_dim_height, outputilm_width, 128) [have to check]
        dL_dinput = dL_dinput.reshape((10, 14, 14, 128))
        print("After reshape", dL_dinput.shape)
        
        # relu activation layer
        dL_dinput = self.relu.backprop(dL_dinput)
        print("After relu backward", dL_dinput.shape)
        
        # third conv layer
        dL_dinput, dL_dweights, dL_dbias = self.conv3.backward(dL_dinput)
        print("After conv3 backward", dL_dinput.shape)
        
        # max pooling layer
        dL_dinput = self.pool1.backward(dL_dinput)
        print("After pool1 backward", dL_dinput.shape)
        
        # first two conv layer
        dL_dinput, dL_dweights, dL_dbias = self.conv2.backward(dL_dinput)
        print("After conv2 backward", dL_dinput.shape)
        
        dL_dinput, dL_dweights, dL_dbias = self.conv1.backward(dL_dinput)
        print("After conv1 backward", dL_dinput.shape)
        
        return dL_dinput

    @property
    def params(self):
        return (self.conv1.params + 
                self.conv2.params + 
                self.conv3.params + 
                self.fc1.params
        )