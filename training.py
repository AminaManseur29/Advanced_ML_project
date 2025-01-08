import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Definition of the training function
def train(model, optimizer, criterion, X_train, y_train, epochs=20, batch_size=64):
    num_samples = X_train.shape[0]
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Mixing data
        perm = torch.randperm(num_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Front and backward passages
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples // batch_size):.4f}")