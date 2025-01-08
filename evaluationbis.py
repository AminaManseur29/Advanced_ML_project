import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the PyTorch model.

    Parameters:
    model: The PyTorch model to be evaluated
    test_loader: DataLoader for the test set

    Returns:
    accuracy (float): The accuracy of the model on the test set
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == y_test).sum().item()
        total += y_test.size(0)

    accuracy = correct / total
    return accuracy
