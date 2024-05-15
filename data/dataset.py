import pandas as pd 
import numpy as np


train_path = 'data/EMNIST/emnist-digits-train.csv'
test_path = 'data/EMNIST/emnist-digits-test.csv'

def loadData(data_path):
    '''
    load data from csv files
    
    args:
    file_path: path to csv file
    
    returns:
    images: arrays containing image and labels 
    '''
    
    data = pd.read_csv(data_path)
    images = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    
    return images, labels

def preprocessData(images, labels, img_height = 28, img_width = 28):
    '''
    preprocess data
    
    args:
    images: array containing images
    labels: array containing labels
    
    returns:
    images: preprocessed images
    labels: preprocessed labels
    '''
    # reshape the images
    images = images.reshape(-1, img_height, img_width, 1).astype('float32')
    # normalize the images
    images /= 255.0
    
    # one hot encode the labels
    num_classes = np.unique(labels).shape[0]
    labels = np.eye(num_classes)
    
    return images, labels

def loadAndPreprocessData(): # only for test
    '''
    Load and preprocess data
    
    output:
    dictionary containing images and labels
    '''
    train_images, train_labels = loadData(train_path)
    test_images, test_labels = loadData(test_path)
    
    train_images, train_labels = preprocessData(train_images, train_labels)
    test_images, test_labels = preprocessData(test_images, test_labels)
    
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

print(loadAndPreprocessData())