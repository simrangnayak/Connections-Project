# NYT Connections Solver Using BERT

This project provides a transformer-model approach to solving the **NYT Connections puzzle**. By leveraging **BERT embeddings**, the model groups words based on semantic similarity and evaluates performance using metrics like pairwise accuracy, F1-score, and specificity. Additionally, epoch loss graphs and confusion matrices provide insights into the model's performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Model Workflow](#model-workflow)
4. [Performance Metrics}](#performance-metrics)
5. [Customizing Visualizations](#customizing-visualizations)
6. [Installation](#installation)

---

## Introduction
The **NYT Connections puzzle** involves grouping words into predefined categories based on thematic similarities. This project aims to predict word groups using BERT embeddings and visualize the results to understand the model's strengths and weaknesses.

## Project Structure
The repository is organized as follows:
- **Code Files**:
  - `model.py`: Defines the BERT-based fine-tuning model and training logic.
  - `helper.py`: Utility functions for data manipulation, including shuffling and contrastive loss function.
  - `testing.py`: Script for evaluating the model.
  - `processing.py`: Handles data preprocessing, including formatting for model input.
  - `plots.py`: Functions to generate and save evaluation plots (e.g., epoch loss, confusion matrices).
  - `cross_validation.py`: Implements cross-validation logic for fine-tuning hyperparameters.
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
   - Fine-tune the model based on contrastive loss, which helps in learning to differentiate between pairs of words that belong to different categories.
   - Use k-fold cross-validation to optimize hyperparameters.
3. **Evaluation**:
   - Use metrics (pairwise weighted accuracy, F1-score, specificity) to evaluate the model.
   - Generate epoch loss graphs and confusion matrices for visualization.

## Performance Metrics
The project evaluates performance using the following:
- **Accuracy**: Weighted accuracy of word pairings
- **F1-Score**: Harmonic mean of precision and recall (note: in this project, precision and recall are symmetric)
- **Specificity**: Proportion of true negatives correctly identified
- **Confusion Matrix**: Breakdown of true/false positives and negatives

Metrics are reported for:
- **Training Set**: Post-training metrics
- **Test Set**:
  - **Baseline Metrics**: Before fine-tuning
  - **Fine-Tuned Metrics**: After fine-tuning the model
  - **Epoch Loss**

### Customizing Visualizations
To adjust model outputs and visualizations:
1. Modify the model configuration in **`model.py`** (e.g., learning rate, batch size) under the # TO CHANGE section.
2. Update cross-validation settings in **`cross_validation.py`** under the # TO CHANGE section.
3. Extend visualizations in **`plots.py`** using logged metrics. Plots will be saved in the `plots/` directory.

## Installation
### Prerequisites
- Python 3.8+
- Required libraries: PyTorch, Transformers, NumPy, Pandas, Scikit-learn, Matplotlib, JSON, Seaborn

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nyt-connections-solver.git
   cd nyt-connections-solver
