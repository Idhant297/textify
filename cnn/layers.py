import numpy as np

# Convolution layer
class Conv2DLayer:
    def __init__(self, input_channel, num_filters, kernel_size, stride = 1, padding = 0):
        self.input_channel = input_channel
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(num_filters, input_channel, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((num_filters, 1))
        self.cache = None

    def forward(self, input):
        '''
        Parameters:
        input: numpy array of shape (batch_size, h, w, channels)
        '''
        print("Input shape:", input.shape)
        
        self.cache = input
        batch_size, h, w, channels = input.shape
        padded_input = np.pad(input,
                              ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                              mode='constant')
        new_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        new_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, new_h, new_w, self.num_filters))

        for i in range(batch_size):
            for j in range(self.num_filters):
                for h_idx in range(new_h):
                    for w_idx in range(new_w):
                        h_start = h_idx * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_idx * self.stride
                        w_end = w_start + self.kernel_size
                        window = padded_input[i, h_start:h_end, w_start:w_end, :]
                        #transpose the weights
                        weight_to_use = self.weights[j].transpose(1, 2, 0)
                        output[i, h_idx, w_idx, j] = np.sum(window * weight_to_use) + self.bias[j]
        self.cache = (input, padded_input)
        return output

    def backward(self, dL_dout):
        '''
        Parameters:
        dL_dout: numpy array of shape (h, w, num_filters)
        
        Returns:
        dL_dinput: numpy array of shape (h, w, num_filters)
        dL_dweights: numpy array of shape (num_filters, input_channel, kernel_size, kernel_size)
        dL_dbias: numpy array of shape (num_filters, 1)
        '''
        input, padded_input = self.cache
        batch_size, h, w, _ = input.shape
        _, h_out, w_out, _ = dL_dout.shape

        # Initialize gradients with zeros
        dL_dinput_padded = np.zeros_like(padded_input)
        dL_dweights = np.zeros_like(self.weights)
        dL_dbias = np.zeros_like(self.bias)

        # # Iterate over each example in the batch
        for i in range(batch_size):
            for j in range(self.num_filters):
                for h_idx in range(h_out):
                    for w_idx in range(w_out):
                        h_start = h_idx * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_idx * self.stride
                        w_end = w_start + self.kernel_size

                        # Calculate gradients
                        window = padded_input[i, h_start:h_end, w_start:w_end, :]
                        window = window.transpose(2, 0, 1)
                        weight_to_use = self.weights[j].transpose(1, 2, 0)
                        dL_dinput_padded[i, h_start:h_end, w_start:w_end, :] += weight_to_use * dL_dout[i, h_idx, w_idx, j]
                        #dL_dinput_padded[i, h_start:h_end, w_start:w_end, :] += self.weights[j] * dL_dout[i, h_idx, w_idx, j]
                        dL_dweights[j] += window * dL_dout[i, h_idx, w_idx, j]
                        dL_dbias[j] += dL_dout[i, h_idx, w_idx, j]

        # Remove padding from the gradient wrt input if there was any
        if self.padding != 0:
            dL_dinput = dL_dinput_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dL_dinput = dL_dinput_padded

        return dL_dinput, dL_dweights, dL_dbias
    
    @property
    def params(self):
        return [self.weights, self.bias]

# ReLU Activation Layer
class ReLULayer:
    def __init__(self):
        self.cache = None

    def forward(self, input):
        self.cache = input
        return np.maximum(0, input)

    def backprop(self, dL_dout):
        input = self.cache
        return dL_dout * (input > 0)

# Fully Connected Layer
class FullyConnectedLayer:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        self.cache = None

    def forward(self, input):
        self.cache = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, dL_dout):
        input = self.cache
        dL_dweights = np.dot(input.T, dL_dout)
        dL_dbias = np.sum(dL_dout, axis = 0)
        dL_dinput = np.dot(dL_dout, self.weights.T)

        lr = 0.01
        self.weights -= lr * dL_dweights
        self.bias -= lr * dL_dbias

        return dL_dinput
    
    @property
    def params(self):
        return [self.weights, self.bias]

# Flatten Layer
class FlattenLayer:
    def __init__(self):
        pass

    def forward(self, input):
        self.cache = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, dL_dout):
        return dL_dout.reshape(self.cache)

# Activation Layers
class activation:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derv(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def tanh(self, x):
        return np.tanh(x)

# Pooling Layer
class MaxPoolingLayer:
    def __init__(self, kernel_size, stride):
        self.pool_size = kernel_size
        self.stride = stride

    def iterate_regions(self, image):
        '''
        Parameters:
        image: numpy array of shape (h, w, num_filters)
        '''
        h, w, _ = image.shape
        new_h = h // self.stride
        new_w = w // self.stride
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.stride):(i * self.stride + self.pool_size), (j * self.stride):(j * self.stride + self.pool_size)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Parameters:
        input: numpy array of shape (h, w, num_filters)
        '''

        self.cache = input
        batch_size, h, w, num_filters = input.shape
        new_h = h // self.stride
        new_w = w // self.stride
        output = np.zeros((batch_size, new_h, new_w, num_filters))

        for i in range(batch_size):
            for h_idx in range(new_h):
                for w_idx in range(new_w):
                    for nF in range(num_filters):
                        h_start = h_idx * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w_idx * self.stride
                        w_end = w_start + self.pool_size
                        window = input[i, h_start:h_end, w_start:w_end, nF]
                        output[i, h_idx, w_idx, nF] = np.max(window)
        return output

    def backward(self, dL_dout):
        '''
        Parameters:
        dL_dout: numpy array of shape (h, w, num_filters)

        Returns:
        dL_dinput: numpy array of shape (h, w, num_filters)
        '''
        input = self.cache
        batch_size, h, w, num_filters = input.shape
        dL_dinput = np.zeros_like(input)

        for i in range(batch_size):
            for nF in range(num_filters):
                for h_idx in range(0, h, self.stride):
                    for w_idx in range(0, w, self.stride):
                        h_start = h_idx
                        h_end = h_idx + self.pool_size
                        w_start = w_idx
                        w_end = w_idx + self.pool_size

                        # Extract the window
                        window = input[i, h_start:h_end, w_start:w_end, nF]
                        max_val = np.max(window)
                        
                        # Calculate the mask
                        mask = (window == max_val)
                        
                        # Update gradients in the input gradient array
                        dL_dinput[i, h_start:h_end, w_start:w_end, nF] += (
                            dL_dout[i, h_idx // self.stride, w_idx // self.stride, nF] * mask
                        )

        return dL_dinput