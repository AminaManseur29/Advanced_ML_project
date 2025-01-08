import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def training_validation(model, optimizer, criterion, X_train, y_train, X_valid, y_valid, 
                        epochs=20, batch_size=64):
    """
    Entraîne et valide un modèle PyTorch sans utiliser de DataLoader.

    Parameters:
    - model: Le modèle PyTorch à entraîner.
    - optimizer: L'optimiseur (par exemple, BGE_Adam).
    - criterion: La fonction de perte (par exemple, CrossEntropyLoss).
    - X_train, y_train: Données et étiquettes pour l'entraînement (PyTorch tensors).
    - X_valid, y_valid: Données et étiquettes pour la validation (PyTorch tensors).
    - epochs: Nombre d'époques d'entraînement.
    - batch_size: Taille des mini-lots.

    Returns:
    - train_losses: Liste des pertes d'entraînement pour chaque époque.
    - val_losses: Liste des pertes de validation pour chaque époque.
    - val_accuracies: Liste des précisions de validation pour chaque époque.
    """
    num_samples = X_train.shape[0]
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()  # Passer en mode entraînement
        epoch_loss = 0
        
        # Mélanger les données d'entraînement
        perm = torch.randperm(num_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Diviser les données en mini-lots
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Réinitialiser les gradients
            optimizer.zero_grad()

            # Passages avant et arrière
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calcul de la perte moyenne pour l'époque
        train_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(train_loss)

        # Validation
        model.eval()  # Passer en mode évaluation
        with torch.no_grad():
            val_outputs = model(X_valid)
            val_loss = criterion(val_outputs, y_valid).item()
            val_losses.append(val_loss)

            # Calcul de la précision
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == y_valid).float().mean().item()
            val_accuracies.append(val_accuracy)

        # Affichage des résultats pour l'époque
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies