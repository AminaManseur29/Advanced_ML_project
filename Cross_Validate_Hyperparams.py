from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn


def cross_validate_hyperparams(model_class, optimizer_class, X_train, y_train, params, input_dim, num_classes, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    best_loss = float('inf')
    best_params = None
    
    with tqdm(total=len(params) * k_folds, desc="Search of the hyperparameters") as pbar:
        for param_set in params:
            fold_losses = []
            for train_idx, val_idx in kfold.split(X_train):
                X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
                X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

                model = model_class(input_dim, num_classes)
                optimizer = optimizer_class(model.parameters(), **param_set)
                criterion = nn.CrossEntropyLoss()

                # Train on training fold
                for epoch in range(5):  # Short training for hyperparameter tuning
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train_fold)
                    loss = criterion(outputs, y_train_fold)
                    loss.backward()
                    optimizer.step()

                # Validate on validation fold
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold)
                    val_loss = criterion(val_outputs, y_val_fold).item()
                    fold_losses.append(val_loss)
                pbar.update(1)

            avg_loss = sum(fold_losses) / len(fold_losses)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = param_set
    
    return best_params, best_loss


def cross_validate_accuracy(model_class, optimizer_class, X_train, y_train, params, input_dim, num_classes, k_folds=10):    
    kfold = KFold(n_splits=k_folds, shuffle=True)
    accuracies = []

    with tqdm(total=k_folds, desc="Cross validation") as pbar:
        for train_idx, val_idx in kfold.split(X_train):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            model = model_class(input_dim, num_classes)
            optimizer = optimizer_class(model.parameters(), **params)
            criterion = nn.CrossEntropyLoss()

            # Train on training fold
            for epoch in range(10):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_fold)
                loss = criterion(outputs, y_train_fold)
                loss.backward()
                optimizer.step()

            # Validate on validation fold
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                val_predictions = torch.argmax(val_outputs, dim=1)
                accuracy = (val_predictions == y_val_fold).float().mean().item()
                accuracies.append(accuracy)
            pbar.update(1)

    return sum(accuracies) / len(accuracies)