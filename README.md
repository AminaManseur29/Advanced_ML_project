# Advanced Machine Learning Project

This GitHub repository contains the source code for the final project of the **Advanced Machine Learning** course taught by Austin Stromme during the 1st Semester of the thrid year at ENSAE Paris.

## Project overview 

Our project explores BGE-Adam, an advanced variant of the Adam optimizer designed to improve adaptability, convergence speed, and robustness in dynamic applications, such as medical image analysis and financial time series forecasting. BGE-Adam incorporates algorithmic innovations to enhance model training under diverse, high-dimensional data conditions, making it particularly valuable in fields with high data variability and complexity.

## Objectives 
1. **Understanding BGE-Adam**: Provide a comprehensive introduction to BGE-Adam, highlighting key innovations that enhance adaptability, convergence speed, and robustness. This section will cover the algorithm’s structure and unique features that differentiate it from the standard Adam optimizer.
2. **Implementation** :  
- We tested the performance of BGE-Adam with different Adam variants (BGE-RMSPro, AdaGrad, and Adaptive Adam) using classic test functions that represent various optimization problems. These include well-conditioned and poorly-conditioned quadratic functions, non-convex functions, and functions with local minima, evaluating BGE-Adam's robustness, efficiency, convergence speed, and stability compared to standard optimizers.
- MNIST dataset: Implement and evaluate BGE-Adam by training models (Logistic Regression, Simple Neural Network, and Complex Neural Network) on the MNIST dataset. The objective is to achieve accurate classification of handwritten digits, providing a controlled environment to assess the performance improvements over existing optimizers (BGE-Adam, Adam, and SGD).

3. **Comparison and analysis**: 
- Performance evaluation: Benchmark BGE-Adam’s performance against Adam and SGD across various models and hyperparameter configurations. Focus will be placed on improvements in accuracy, stability, and adaptability, especially under variable and challenging training conditions.
- Model complexity and optimization: Examine how BGE-Adam handles different model complexities and how it compares in terms of convergence speed and robustness, particularly on high-dimensional data.

## Contents

### Folders

- **/Models**: This folder contains modules related to models and optimization:
  - `BGE_Adam.py`: Implementation of the BGE_Adam optimizer.
  - `evaluation.py`: Functions for evaluating models (accuracy, loss, etc.).
  - `models.py`: Definitions of models used (logistic regression, simple neural networks, complex networks).
  - `training_validation.py`: Functions for model training and validation.

- **/Utils**: This folder contains utility tools:
  - `Cross_Validate_Hyperparams.py`: Hyperparameter cross-validation to optimize model performance.
  - `utils.py` : 

- **Theoretical_Experiments.ipynb`** : This notebook presents empirical experiments on the Sentic dataset, where we implemented Adam variants (BGE-RMSPro, AdaGrad, and Adaptive Adam) and tested the performance of BGE-Adam using some classic test functions.
- **`Empirical_experiments.ipynb`**: This notebook presents empirical experiments conducted, illustrating different hyperparameter validation methods and model training approaches. It demonstrates how to reproduce the empirical results obtained, with sections on data preparation, K-Fold Cross Validation, and training with optimal hyperparameters.

- **`Theoretical_experiments.ipynb`**: This notebook documents theoretical experiments, where we seek to minimize various functions by exploring different optimization configurations using algorithms such as BGE_Adam.

- **README.md**: This file.

- **`requirements.txt`**: List of dependencies required for the project.

## Setup Instructions

From the command line, follow these steps to set up this project:

1. Clone this repository:
```bash
   git clone https://github.com/AminaManseur29/Advanced_ML_project.git
```
2. Navigate to the project folder:
```bash
cd Advanced_ML_project
```
3. Install the listed dependencies:
```bash
pip install -r requirements.txt
```

## Contact

| Name            | Email                |
|----------------|----------------------|
| Oussama Es Semyry    | oussama.es-semyry@ensae.fr |
| Louise Ligonnière  | louise.ligonniere@ensae.fr |
| Amina Manseur   | amina.manseur@ensae.fr |
