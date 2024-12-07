# NYT Connections Solver Using BERT

This project provides a transformer-model approach to solving the **NYT Connections puzzle**. By leveraging **BERT embeddings**, the model groups words based on semantic similarity and evaluates performance using metrics like pairwise accuracy, F1-score, and specificity. Additionally, epoch loss graphs and confusion matrices provide insights into the model's performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Model Workflow](#model-workflow)
4. [Metrics and Evaluation](#metrics-and-evaluation)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction
The **NYT Connections puzzle** involves grouping words into predefined categories based on thematic similarities. This project aims to predict word groups using BERT embeddings and visualize the results to understand the model's strengths and weaknesses.

## Project Structure
The repository is organized as follows:
- **Code Files**:
  - `model.py`: Contains the BERT model definition and training logic.
  - `helper.py`: Utility functions for data manipulation and evaluation.
  - `testing.py`: Script for evaluating the model on test puzzles.
  - `processing.py`: Preprocessing scripts for data preparation.
  - `plots.py`: Functions to generate and save evaluation plots (e.g., epoch loss, confusion matrices).
  - `cross_validation.py`: Implements k-fold cross-validation for hyperparameter tuning.
- **Data**:
  - `connections.json`: The main dataset containing puzzle words and their correct groupings.
- **Outputs**:
  - `plots/`: Directory to store generated plots such as confusion matrices, precision/recall trends, and more.
  - Miscellaneous evaluation artifacts like logs and result files.

## Model Workflow
1. **Data Preprocessing**: 
   - Extract word embeddings from a pre-trained **BERT model**.
   - Organize data for training, validation, and testing.
2. **Training**:
   - Fine-tune the model based on contrastive loss
   - Use k-fold cross-validation to optimize hyperparameters.
3. **Evaluation**:
   - Use metrics (pairwise accuracy, F1-score, specificity) to evaluate the model.
   - Generate epoch loss graphs and confusion matrices for visualization.

## Metrics and Evaluation
The project evaluates performance using word pairings:
- **Accuracy**: Weighted accuracy of word pairings
- **F1-Score**: Harmonic mean of precision and recall (note: in this project, precision, recall, and F1-score are all symmetric).
- **Specificity**: Proportion of true negatives correctly identified.
- **Confusion Matrix**: Breakdown of true/false positives and negatives.

Visualizations include:
- Epoch loss graphs
- Confusion matrix heatmaps

## Installation
### Prerequisites
- Python 3.8+
- Required libraries: PyTorch, NumPy, Scikit-learn, Matplotlib, JSON

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nyt-connections-solver.git
   cd nyt-connections-solver
