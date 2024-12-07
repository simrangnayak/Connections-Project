import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# transformer outputs
epochs = [1, 2, 3, 4, 5, 6, 7, 8]  # Epoch numbers
bert_baseline = [2910.7571826, 1816.65516281, 1056.31808496, 685.85971719, 501.88458085, 397.37958193, 338.59407294, 391.1240668]
roberta_baseline = [3446.3022604, 3032.87777758, 2454.54371691, 1829.59558129, 1335.41902828, 999.48249733, 777.6423434, 645.70610571]
distilbert_baseline = [2856.62018442, 1705.65068221, 988.34845543, 661.29773545, 493.68199661, 406.79008976, 352.9943893, 307.99746877]

# depth outputs
bert_single_layer = [3216.83072138, 3064.27454996, 2961.32044363, 2876.70613575, 2797.90174007, 2725.54354239, 2642.77698994, 2558.93472528]
bert_three_layer = [3194.63259602, 2921.02111244, 2693.3029213, 2464.85849643, 2223.49712181, 1997.9282167, 1782.68994999, 1590.24513578]

# optimal outputs
bert_optimal = [3253.13976145, 3066.71469212, 2969.51328945, 2889.37826443, 2815.3724103, 2713.63315344, 2651.04436207, 2570.55659628]

# Create the transformers variant plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, bert_baseline, marker='o', linestyle='-', color='b', label='BERT')
plt.plot(epochs, roberta_baseline, marker='o', linestyle='-', color='r', label='RoBERTa')
plt.plot(epochs, distilbert_baseline, marker='o', linestyle='-', color='g', label='DistilBERT')
plt.title('Contrastive Loss per Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)  # Ensure x-axis ticks align with epochs
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join('plots', 'transformer_models.png'))

# Create different depths plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, bert_single_layer, marker='o', linestyle='-', color='b', label='Scenario 1')
plt.plot(epochs, bert_three_layer, marker='o', linestyle='-', color='g', label='Scenario 2')
plt.title('Contrastive Loss for Different Model Depths', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)  # Ensure x-axis ticks align with epochs
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join('plots', 'bert_layers.png'))

# Create the optimal BERT plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, bert_optimal, marker='o', linestyle='-', color='b', label='Optimal LR and WD')
plt.title('Optimal BERT Hyper-parameters', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)  # Ensure x-axis ticks align with epochs
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join('plots', 'optimal_bert.png'))

# Base confusion matrix values
base_TP = 807
base_FP = 2745
base_FN = 2745
base_TN = 11463

# After fine-tuning confusion matrix values
post_TP = 1258
post_FP = 2294
post_FN = 2294
post_TN = 11914

# Create the confusion matrices
base_confusion_matrix = np.array([[base_TP, base_FP], [base_FN, base_TN]])
post_confusion_matrix = np.array([[post_TP, post_FP], [post_FN, post_TN]])

# Define labels
labels = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives']
categories = ['Positive', 'Negative']

# Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(base_confusion_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=categories, yticklabels=categories,
            annot_kws={"size": 16})  # Increase the font size for the numbers

# Add titles and labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.savefig(os.path.join('plots', 'baseline_cm.png'))

# Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(post_confusion_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=categories, yticklabels=categories,
            annot_kws={"size": 16})  # Increase the font size for the numbers

# Add titles and labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.savefig(os.path.join('plots', 'post_cm.png'))