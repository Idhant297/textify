import numpy as np
import pandas as pd
import pickle
import cnn.model as model

def loadData():
    '''
    Load the MNIST dataset
    '''
    # Load the dataset
    train = pd.read_csv('data/EMNIST/emnist-mnist-train.csv')
    test = pd.read_csv('data/EMNIST/emnist-mnist-test.csv')
    
    # Extract the data
    train_data = train.iloc[:, 1:].values
    train_labels = train.iloc[:, 0].values
    test_data = test.iloc[:, 1:].values
    test_labels = test.iloc[:, 0].values
    
    # Reshape the data
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    
    return train_data, train_labels, test_data, test_labels

def one_hot_encode(labels, num_classes):
    '''
    labels (numpy array) : array of labels
    num_classes (int) : number of classes
    
    Returns:
    one_hot (numpy array) : one hot encoding of the labels
    '''
    
    one_hot_encode = np.zeros((labels.shape[0], num_classes))
    one_hot_encode[np.arange(labels.shape[0]), labels] = 1
    return one_hot_encode

def computeLoss(predictions, labels):
    '''
    Compute the cross entropy loss
    
    Parameters:
    predictions (numpy array) : array of predictions
    labels (numpy array) : array of labels
    
    Returns:
    loss (float) : cross entropy loss
    '''
    
    num_samples = predictions.shape[0]
    loss = -np.sum(labels * np.log(predictions)) / num_samples
    return loss

def softmax(x):
    '''
    Compute softmax values for each sets of scores in x.
    
    Parameters:
    x (numpy array) : array of scores
    
    Returns:
    softmax (numpy array) : softmax of the scores
    '''
    
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def saveModel(model, path):
    '''
    Save the trained model
    
    Parameters:
    model : trained model
    path : path to save the model
    
    '''
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    
def shuffleData(images, labels):
    '''
    Parameters:
    images (numpy array) : array of images
    labels (numpy array) : array of labels
    
    Returns:
    images (numpy array) : shuffled array of images
    labels (numpy array) : shuffled array of labels
    '''
    # shuffle the data
    permutation = np.random.permutation(images.shape[0])
    images = images[permutation]
    labels = labels[permutation]
    
    return images, labels
    
    
# def loadModel(filename):
#     """
#     Load a model from a file.
    
#     Args:
#         filename (str): The path to the file containing the model's saved parameters.
    
#     Returns:
#         model (object): An instance of the model with loaded parameters.
#     """
#     with open(filename, 'rb') as file:
#         model_parameters = pickle.load(file)

#     # Assuming CNN is your model class, and it has a method to initialize with parameters
#     model_ = model.CNN()
#     model_.(model_parameters)
#     return model_