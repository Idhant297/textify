# Implement forward propagation.

from model import CNN
import pandas as pd
import numpy as np

def forwardPass(input_data, model):
    '''
    Forward Pass through the CNN model
    
    Parameters:
    input_data: the data to the cnn (images)
    model: the cnn model
    
    Output:
    output: output from the final layer of the cnn model
    '''
    return model.forward(input_data)

def load_emnist(file_path):
    """ Load EMNIST data from a csv file """
    data = pd.read_csv(file_path)
    # Assuming the last column is the label and the rest are pixel values
    images = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Reshape the images to 28x28 and normalize
    images = images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0

    # Optionally: Convert labels to one-hot encoding
    labels = np.eye(26)[labels]  # Adjust the 26 based on number of classes, 10 for digits, 26 for letters, etc.

    return images, labels

# Example usage:
train_images, train_labels = load_emnist('data/EMNIST/emnist-mnist-train.csv')
input_data = train_images[:10]  # Use the first 10 images for example

if __name__ == "__main__":
    # Example input data, e.g., a batch of images. Assume it's loaded and preprocessed correctly
    input_data = train_images[:10]

    # Initialize the model
    cnn_model = CNN()

    # Perform the forward pass
    output = forwardPass(input_data, cnn_model)
    print("Output of the forward pass:", output)

