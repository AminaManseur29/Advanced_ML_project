# Advanced Machine Learning Project

This GitHub repository contains the source code for the final project of the **Advanced Machine Learning** course taught by Austin Stromme during the 1st Semester of the thrid year at ENSAE Paris.

## Contents

### Folders

- **/Experiments**: This folder contains two Jupyter Notebooks:
  - `Empirical_experiments.ipynb`: This notebook presents empirical experiments conducted, illustrating different hyperparameter validation methods and model training approaches. It demonstrates how to reproduce the empirical results obtained, with sections on data preparation, K-Fold Cross Validation, and training with optimal hyperparameters.
  - `Theoretical_experiments.ipynb`: This notebook documents theoretical experiments, where we seek to minimize various functions by exploring different optimization configurations using algorithms such as BGE_Adam.

- **/Models**: This folder contains modules related to models and optimization:
  - `BGE_Adam.py`: Implementation of the BGE_Adam optimizer.
  - `evaluation.py`: Functions for evaluating models (accuracy, loss, etc.).
  - `models.py`: Definitions of models used (logistic regression, simple neural networks, complex networks).
  - `training_validation.py`: Functions for model training and validation.

- **/Utils**: This folder contains utility tools:
  - `Cross_Validate_Hyperparams.py`: Hyperparameter cross-validation to optimize model performance.
  - `utils.py` : 

- **README.md**: This file.

- **requirements.txt**: List of dependencies required for the project.

## Setup Instructions

From the command line, follow these steps to set up this project:

1. Clone this repository:
```bash
   git clone https://github.com/YourUsername/YourProject.git
```
2. Navigate to the project folder:
```bash
cd YourProject
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