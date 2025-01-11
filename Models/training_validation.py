import torch
import matplotlib.pyplot as plt
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


# Definition of the training and validation function
def training_validation(model, optimizer, criterion, X_train, y_train, X_valid, y_valid, 
                        epochs=20, batch_size=64):
    """
    Train and validate a PyTorch model without using a DataLoader.

    Parameters:
    - model: The PyTorch model to be trained.
    - optimizer: The optimizer (e.g., BGE_Adam).
    - criterion: The loss function (e.g., CrossEntropyLoss).
    - X_train, y_train: Training data and labels (PyTorch tensors).
    - X_valid, y_valid: Validation data and labels (PyTorch tensors).
    - epochs: Number of training epochs.
    - batch_size: Batch size.

    Returns:
    - train_losses: List of training losses for each epoch.
    - val_losses: List of validation losses for each epoch.
    - val_accuracies: List of validation accuracies for each epoch.
    """    

    num_samples = X_train.shape[0]
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()  # Training mode
        epoch_loss = 0
        
        # Shuffle training data
        perm = torch.randperm(num_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Divide the training data into mini-batches
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Reset gradients
            optimizer.zero_grad()

            # Forward and backward
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate the average loss for the epoch
        train_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(train_loss)

        # Validation
        model.eval()  # Evaluation mode
        with torch.no_grad():
            val_outputs = model(X_valid)
            val_loss = criterion(val_outputs, y_valid).item()
            val_losses.append(val_loss)

            # Calculate accuracy
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == y_valid).float().mean().item()
            val_accuracies.append(val_accuracy)

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies


# Definition of the Trainer class
class Trainer:
    def __init__(self, model_class, input_dim, num_classes, criterion, optimizers, epochs=20, batch_size=64):
        """
    Initialize the Trainer class.

    Parameters:
    - model_class: The model class (e.g., LogisticRegressionTorch, SimpleNN, etc.).
    - input_dim: Dimension of the input data.
    - num_classes: Number of output classes.
    - criterion: The loss function.
    - optimizers: Dictionary of optimizers to compare (name -> class).
    - epochs: Number of training epochs.
    - batch_size: Batch size.
    """

        self.model_class = model_class
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizers = optimizers
        self.epochs = epochs
        self.batch_size = batch_size

    def train_and_validate(self, model, optimizer, X_train, y_train, X_valid, y_valid):
        """
    Train and validate a model.

    Returns:
    - train_losses, val_losses, val_accuracies: Lists of training and validation metrics.
    """
        
        num_samples = X_train.shape[0]
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(self.epochs):
            model.train() # Training mode
            epoch_loss = 0

            # Shuffle training data
            perm = torch.randperm(num_samples)
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Training by mini-batches
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                # Reset gradients
                optimizer.zero_grad()

                # Forward and backward
                outputs = model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss for the epoch
            train_loss = epoch_loss / (num_samples // self.batch_size)
            train_losses.append(train_loss)

            # Validation
            model.eval()    # Evaluation mode
            with torch.no_grad():
                val_outputs = model(X_valid)
                val_loss = self.criterion(val_outputs, y_valid).item()
                val_losses.append(val_loss)

                # Accuracy
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_predictions == y_valid).float().mean().item()
                val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return train_losses, val_losses, val_accuracies

    def compare_optimizers(self, X_train, y_train, X_valid, y_valid, hyperparams_results):
        """
        Compare optimizers in terms of convergence, stability, and accuracy.

        Returns:
        - results: Dictionary containing metrics for each optimizer.
        """
        results = {}

        for name, opt_class in self.optimizers.items():
            print(f"\nRunning {name} optimizer...")
            model = self.model_class(self.input_dim, self.num_classes)

            if name == "BGE_Adam":
                params = {'lr':0.01}
            else:
                params = hyperparams_results[f'{model}_{name}']

            optimizer = opt_class(model.parameters(), **params)

            train_losses, val_losses, val_accuracies = self.train_and_validate(
                model, optimizer, X_train, y_train, X_valid, y_valid
            )

            results[name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            }

        self.plot_results(results)
        return results

    def plot_results(self, results):
        """
        Plot loss and accuracy curves for each optimizer.
        """
        plt.figure(figsize=(14, 6))

        # Train Losses
        plt.subplot(1, 2, 1)
        for name, metrics in results.items():
            plt.plot(metrics['train_losses'], label=name)
        plt.title("Train Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Validation Accuracy
        plt.subplot(1, 2, 2)
        for name, metrics in results.items():
            plt.plot(metrics['val_accuracies'], label=name)
        plt.title("Validation Accuracies")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
