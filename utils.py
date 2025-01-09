import time
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from models import LogisticRegressionTorch, SimpleNN, ComplexNN
from BGE_Adam import BGE_Adam
import numpy as np

# Util 1 : comparison of optimizers for loss convergence, stability, gradient norms and parameter updates
def run_optimizer(optimizer_class, model, X_train, y_train, criterion, lr=0.01, epochs=10, batch_size=64, **kwargs):
    optimizer = optimizer_class(model.parameters(), lr=lr, **kwargs)
    losses = []
    times = []
    grad_norms = []
    param_updates = []
    prev_params = [p.clone().detach() for p in model.parameters()]
    start_time = time.time()
    accumulated_time = 0

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_train.size(0))
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            iteration_start = time.time()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            iteration_end = time.time()
            iteration_time = iteration_end - iteration_start
            accumulated_time += iteration_time

            # Collect metrics
            with torch.no_grad():
                current_grad_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in model.parameters())).item()
                grad_norms.append(current_grad_norm)

                update_norm = torch.sqrt(sum(torch.sum((p - pp)**2) for p, pp in zip(model.parameters(), prev_params))).item()
                param_updates.append(update_norm)
                prev_params = [p.clone().detach() for p in model.parameters()]

            losses.append(loss.item())
            times.append(accumulated_time)

    end_time = time.time()
    return losses, times, grad_norms, param_updates, end_time - start_time

def compare_optimizers(model_class, input_dim, num_classes, X_train, y_train, X_test, y_test):
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'BGE_Adam': BGE_Adam
    }
    
    results = {}
    criterion = torch.nn.CrossEntropyLoss()
    
    for name, opt_class in optimizers.items():
        print(f'Running {name}...')
        model = model_class(input_dim, num_classes)
        losses, times, grad_norms, param_updates, elapsed_time = run_optimizer(
            opt_class, model, X_train, y_train, criterion, lr=0.01, epochs=10, batch_size=64
        )
        results[name] = {
            'losses': losses,
            'times': times,
            'grad_norms': grad_norms,
            'param_updates': param_updates,
            'time': elapsed_time
        }
    
    # Plotting results
    plt.figure(figsize=(14, 10))
    
    # Convergence speed (Loss vs. Time)
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['times'], result['losses'], label=f'{name} ({result["time"]:.2f}s)')
    plt.title('Convergence Speed (Loss vs. Time)')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.legend()
    
    # Stability
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        plt.plot(np.log(np.diff(result['losses'])**2), label=f'{name}')
    plt.title('Log Variance of Loss Changes')
    plt.xlabel('Iterations')
    plt.ylabel('Log Variance of Loss Change')
    plt.legend()

    # Gradient norms
    plt.subplot(2, 2, 3)
    for name, result in results.items():
        plt.plot(result['grad_norms'], label=f'{name}')
    plt.title('Gradient Norms')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.legend()
    
    # Parameters updating
    plt.subplot(2, 2, 4)
    for name, result in results.items():
        plt.plot(result['param_updates'], label=f'{name}')
    plt.title('Parameter Updates')
    plt.xlabel('Iterations')
    plt.ylabel('Update Norm')
    plt.legend()

    plt.tight_layout()
    plt.show()