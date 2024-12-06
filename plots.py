import matplotlib.pyplot as plt

# Example data
epochs = [1, 2, 3, 4, 5, 6, 7, 8]  # Epoch numbers
bert_baseline = [2910.7571826, 1816.65516281, 1056.31808496, 685.85971719, 501.88458085, 397.37958193, 338.59407294, 391.1240668]
roberta_baseline = [3446.3022604, 3032.87777758, 2454.54371691, 1829.59558129, 1335.41902828, 999.48249733, 777.6423434, 645.70610571]
distilbert_baseline = [2856.62018442, 1705.65068221, 988.34845543, 661.29773545, 493.68199661, 406.79008976, 352.9943893, 307.99746877]

bert_single_layer = [3216.83072138, 3064.27454996, 2961.32044363, 2876.70613575, 2797.90174007,
 2725.54354239, 2642.77698994, 2558.93472528]
bert_three_layer = [3194.63259602, 2921.02111244, 2693.3029213, 2464.85849643, 2223.49712181,
 1997.9282167,  1782.68994999, 1590.24513578]

bert_optimal = [3253.13976145, 3066.71469212, 2969.51328945, 2889.37826443, 2815.3724103, 2713.63315344, 2651.04436207, 2570.55659628]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, bert_optimal, marker='o', linestyle='-', color='b', label='Optimal LR and WD')
#plt.plot(epochs, bert_three_layer, marker='o', linestyle='-', color='g', label='Scenario 2')
plt.title('Optimal BERT Hyper-parameters', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)  # Ensure x-axis ticks align with epochs
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('optimal_bert.png')



