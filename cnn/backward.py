from model import CNN
import pandas as pd
import numpy as np


def backwardPass(dL_dout, cache):
    '''
    Backward Pass through the CNN model
    
    Parameters:
    dL_dout: upstream gradient
    cache: tuple of (inpt, weight, bias, stride, padding) as stored furing the forward pass
    
    Output:
    dL_dinput: gradient with respect to the input
    dL_dweights: gradient with respect to the weights
    dL_dbias: gradient with respect to the bias
    '''
    input, weights, bias, stride, padding = cache
    N, C, H, W = input.shape
    F, _, HH, WW = weights.shape
    _, _, H_out, W_out = dL_dout.shape
    
    #intialising the gradients
    dL_dinput = np.zeros_like(input)
    dL_dweights = np.zeros_like(weights)
    dL_dbias = np.zeros_like(bias)
    
    #padding the input
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    padded_input = np.pad(input, pad_width, mode='constant')
    padded_dL_dinput = np.pad(dL_dinput, pad_width, mode='constant', constant_values=0)
    
    # backwar pass
    for n in range(N):
        for f in range(F):
            dL_dbias[f] += np.sum(dL_dout[n, f])
            for i in range(H_out):
                for j in range(W_out):
                    current_i = i * stride
                    current_j = j * stride
                    window = padded_input[n, :, current_i:current_i+HH, current_j:current_j+WW]
                    dL_dweights[f] += window * dL_dout[n, :, i, j], np.newaxis()
                    padded_dL_dinput[n, f, current_i:current_i+HH, current_j:current_j+WW] += weights[f] * dL_dout[n, f, i, j]
    
    # remove padding from dL_dinput
    if padding > 0:
        dL_dinput = padded_dL_dinput[:, :, padding:-padding, padding:-padding]
    else:
        dL_dinput = padded_dL_dinput

    return dL_dinput, dL_dweights, dL_dbias


# relu backward pass
def ReLUBackward(dL_dout, cache):
    '''
    Backward Pass for the ReLU layer
    
    Parameters:
    dL_dout: upstream gradient
    cache: the input to the ReLU layer
    
    Output:
    dL_dinput: gradient with respect to the input
    '''
    dL_dinput = dL_dout * (cache > 0)
    return dL_dinput

# maxpooling backward pass
def MaxPoolBackward(dL_dout, cache):
    '''
    Backward Pass for the MaxPooling layer
    
    Parameters:
    dL_dout: upstream gradient
    cache: the input to the MaxPooling layer
    
    Output:
    dL_dinput: gradient with respect to the input
    '''
    input, pool_size, stride = cache
    N, C, H, W = input.shape
    _, _, H_out, W_out = dL_dout.shape
    
    dL_dinput = np.zeros_like(input)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    current_i = i * stride
                    current_j = j * stride
                    window = input[n, c, current_i:current_i+2, current_j:current_j+2]
                    max_val = np.max(window)
                    for k in range(pool_size):
                        for l in range(pool_size):
                            if window[k, l] == max_val:
                                dL_dinput[n, c, current_i+k, current_j+l] = dL_dout[n, c, i, j]
    return dL_dinput

def FullyConnectedBackward(dL_dout, cache):
    '''
    Backward Pass for the Fully Connected layer
    
    Parameters:
    dL_dout: upstream gradient
    cache: the input to the Fully Connected layer
    
    Output:
    dL_dinput: gradient with respect to the input
    dL_dweights: gradient with respect to the weights
    dL_dbias: gradient with respect to the bias
    '''
    input, weights, bias = cache
    dL_dinput = np.dot(dL_dout, weights.T)
    dL_dweights = np.dot(input.T, dL_dout)
    dL_dbias = np.sum(dL_dout, axis=0)

    return dL_dinput, dL_dweights, dL_dbias

# to test backpass (backward.py), you can use forward.py to generate the forward pass cache and then use this function to compute the backward pass.
# you can also use the functions above to compute the backward pass for the ReLU and MaxPooling layers.

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def load_emnist():
    """ Load EMNIST data from a csv file """
    data = pd.read_csv('/Users/idhantgulati/Documents/handwriting_rec/data/EMNIST/emnist-digits-train.csv')
    # Assuming the last column is the label and the rest are pixel values
    images = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Reshape the images to 28x28 and normalize
    images = images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0

    # Optionally: Convert labels to one-hot encoding
    labels = np.eye(26)[labels]  # Adjust the 26 based on number of classes, 10 for digits, 26 for letters, etc.
    print("Data loaded successfully!")
    return images, labels

def compute_loss(outputs, labels):
    """Compute categorical cross-entropy loss."""
    epsilon = 1e-8  # Small constant to avoid log(0)
    return -np.sum(labels * np.log(outputs + epsilon)) / labels.shape[0]


if __name__ == "__main__":
    # Load data
    images, labels = load_emnist()
    input_data = images[:10]  # Use the first 10 images for example
    labels = labels[:10]

    # Initialize the model
    cnn_model = CNN()  # Make sure CNN class has forward and backward methods

    # Perform forward pass
    outputs = cnn_model.forward(input_data)
    print("forward pass done!")
    
    outputs = softmax(outputs)  # Apply softmax to get probabilities
    print("outputs calculated!")
    
    # Compute loss
    loss = compute_loss(outputs, labels)
    print("Computed Loss:", loss)

    # Compute gradients for backpropagation
    dL_dout = (outputs - labels) / labels.shape[0]  # Derivative of loss w.r.t. output
    print("Gradients calculated!")
    
    # Perform backward pass
    gradients = cnn_model.backward(dL_dout)
    print("gathered gradients", gradients)
    print("Gradients computed during the backward pass:", gradients)