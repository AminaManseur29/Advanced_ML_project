import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Definition of the evaluation function
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean().item()
    return accuracy

# Definition of evaluation function
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

# Definition of evaluation function with additional metrics
def evaluate_model_(model, X_test, y_test):
    """
    Evaluate the PyTorch model.

    Parameters:
    model: The PyTorch model to be evaluated
    X_test: Features of the test set
    y_test: True labels of the test set

    Returns:
    accuracy (float): The accuracy of the model on the test set
    precision (float): The precision of the model on the test set
    recall (float): The recall of the model on the test set
    f1_score (float): The F1 score of the model on the test set
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    model.eval()
    correct = 0
    total = 0
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1_score = []

    with torch.no_grad():
        outputs_test = model(X_test)
        _, y_pred = torch.max(outputs_test, 1)
        
        accuracy_val = accuracy(y_test, y_pred)
        precision_val = precision(y_test, y_pred)
        recall_val = recall(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred)

        all_accuracy.append(accuracy_val)
        all_precision.append(precision_val)
        all_recall.append(recall_val)
        all_f1_score.append(f1_val)

    return sum(all_accuracy) / len(all_accuracy), \
           sum(all_precision) / len(all_precision), \
           sum(all_recall) / len(all_recall), \
           sum(all_f1_score) / len(all_f1_score)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1_score_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

# def f1_score_metric(y_true, y_pred):
    # return f1_score(y_true.numpy(), y_pred.numpy(), average='weighted', zero_division=1)

# def recall(y_true, y_pred):
    # return recall_score(y_true.numpy(), y_pred.numpy(), average='weighted', zero_division=1)

# def precision(y_true, y_pred):
    # return precision_score(y_true.numpy(), y_pred.numpy(), average='weighted', zero_division=1)