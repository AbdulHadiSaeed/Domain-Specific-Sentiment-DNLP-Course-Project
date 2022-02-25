import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


score_accuracy = [0.083, 0.112, 0.138, 0.182, 0.275, 0.331, 0.417, 0.433]
score_accuracy_std = [0.030, 0.043, 0.057, 0.08, 0.08, 0.127, 0.122, 0.095]
polarity_accuracy = [0.522, 0.542, 0.541, 0.552, 0.566, 0.573, 0.6, 0.601]
polarity_accuracy_std = [0.1, 0.013, 0.016, 0.024, 0.022, 0.036, 0.035, 0.043]
freq_values = [10, 25, 50, 100, 250, 500, 1000, 2000]

score_accuracy_fixed = [0.321, 0.326, 0.338, 0.356, 0.363, 0.368, 0.370]
score_accuracy_fixed_std = [0.076, 0.079, 0.085, 0.094, 0.096, 0.094]
polarity_accuracy_fixed = [0.566, 0.541, 0.552, 0.56, 0.566, 0.573, 0.6]
polarity_accuracy_fixed_std = [0.024, 0.025, 0.029, 0.030, 0.027, 0.024]
freq_values_fixed = [10, 25, 50, 100, 250, 500, 750]




modified_df = pd.read_csv('../outputs/temp.csv')

sns.lineplot(x='Frequency', y='Test', hue='Type', ci='sd', data=modified_df)
plt.ylim(0, 1)
plt.show()



fig = plt.figure(figsize=(12, 2))
ax1 = fig.add_subplot(121)
ax1.plot(freq_values, score_accuracy)
# ax1.plot(freq_values, polarity_accuracy, label='Polarity Accuracy', linestyle='--')
plt.grid(linestyle='dotted')
plt.xlabel('Frequency', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.title('(a) Evenly sampled train/validation/test split', fontsize=16)
plt.legend(fontsize=13)


ax1 = fig.add_subplot(122)
ax1.plot(freq_values_fixed, score_accuracy_fixed)
# ax1.plot(freq_values, polarity_accuracy, label='Polarity Accuracy', linestyle='--')
plt.grid(linestyle='dotted')
plt.xlabel('Frequency', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.title('(b) Frequency-wise train/validation split, Fixed test split', fontsize=16)
plt.legend(fontsize=13)
plt.show()
