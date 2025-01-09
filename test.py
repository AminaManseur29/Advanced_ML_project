import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model_class, input_dim, num_classes, criterion, optimizers, epochs=20, batch_size=64):
        """
        Initialise la classe Trainer.

        Parameters:
        - model_class: La classe du modèle (ex. LogisticRegressionTorch, SimpleNN, etc.).
        - input_dim: Dimension des entrées.
        - num_classes: Nombre de classes de sortie.
        - criterion: La fonction de perte.
        - optimizers: Dictionnaire des optimiseurs à comparer (nom -> classe).
        - epochs: Nombre d'époques d'entraînement.
        - batch_size: Taille des mini-lots.
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
        Entraîne et valide un modèle.

        Returns:
        - train_losses, val_losses, val_accuracies: Listes des métriques d'entraînement et de validation.
        """
        num_samples = X_train.shape[0]
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(self.epochs):
            model.train() # Mode entraînement
            epoch_loss = 0

            # Mélange aléatoire des données
            perm = torch.randperm(num_samples)
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Entraînement par mini-lots
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                # Réinitialisation des gradients
                optimizer.zero_grad()

                # Forward and backward
                outputs = model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Calcul de la perte moyenne pour l'époque
            train_loss = epoch_loss / (num_samples // self.batch_size)
            train_losses.append(train_loss)

            # Validation
            model.eval()    # Mode évaluation
            with torch.no_grad():
                val_outputs = model(X_valid)
                val_loss = self.criterion(val_outputs, y_valid).item()
                val_losses.append(val_loss)

                # Calcul de la précision
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_predictions == y_valid).float().mean().item()
                val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return train_losses, val_losses, val_accuracies

    def compare_optimizers(self, X_train, y_train, X_valid, y_valid):
        """
        Compare les optimiseurs en termes de convergence, stabilité et précision.

        Returns:
        - results: Dictionnaire contenant les métriques de chaque optimiseur.
        """
        results = {}

        for name, opt_class in self.optimizers.items():
            print(f"\nRunning {name} optimizer...")
            model = self.model_class(self.input_dim, self.num_classes)
            optimizer = opt_class(model.parameters(), lr=0.01)

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
        Trace les courbes des pertes et précisions pour chaque optimiseur.
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