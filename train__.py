import numpy as np
import time
import cnn.model as model
import cnn.layers as layers
import cnn.forward as forward
import cnn.backward as backward
import data.dataset as dataset
import cnn.utils as utils

def train(train_path, epochs, batch_size, learning_rate, save_path):
    # Load and preprocess the data
    train_images, train_labels = dataset.loadData(train_path)
    print(f"Loaded data shapes - Images: {train_images.shape}, Labels: {train_labels.shape}")
    
    train_images, train_labels = dataset.preprocessData(train_images, train_labels)
    print(f"Preprocessed data shapes - Images: {train_images.shape}, Labels: {train_labels.shape}")
    
    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError(f"Number of images ({train_images.shape[0]}) does not match number of labels ({train_labels.shape[0]})")
    
    # Initialize the model
    cnnModel = model.CNN()
    
    num_samples = train_images.shape[0]
    num_batches = num_samples // batch_size
    
    print("Starting to train the model...")
    print(f"Number of samples: {num_samples}, Number of batches: {num_batches}")
    
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        epoch_loss = 0
        
        # Shuffle the data
        indices = np.random.permutation(num_samples)
        train_images = train_images[indices]
        train_labels = train_labels[indices]
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            batch_images = train_images[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            print(f"Batch {batch+1} shapes - Images: {batch_images.shape}, Labels: {batch_labels.shape}")
            
            # Forward pass
            outputs = cnnModel.forward(batch_images)
            outputs = utils.softmax(outputs)
            
            # Compute loss
            loss = utils.computeLoss(outputs, batch_labels)
            epoch_loss += loss
            
            # Backward pass
            dL_dout = (outputs - batch_labels) / batch_size
            cnnModel.backward(dL_dout)
            
            # Update parameters
            for layer in [cnnModel.conv1, cnnModel.conv2, cnnModel.conv3, cnnModel.fc1]:
                if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                    if hasattr(layer, 'dL_dweights'):
                        layer.weights -= learning_rate * layer.dL_dweights
                    if hasattr(layer, 'dL_dbias'):
                        layer.bias -= learning_rate * layer.dL_dbias.reshape(layer.bias.shape)
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/num_batches:.4f}')
        
        # Save the model after each epoch
        utils.saveModel(cnnModel, f"{save_path}_epoch_{epoch+1}.pkl")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return cnnModel

if __name__ == "__main__":
    # Define parameters
    train_path = 'data/EMNIST/emnist-mnist-train.csv'
    epochs = 1
    batch_size = 128
    learning_rate = 0.001
    save_path = 'model/trained_model'
    
    # Call the train function
    trained_model = train(train_path, epochs, batch_size, learning_rate, save_path)