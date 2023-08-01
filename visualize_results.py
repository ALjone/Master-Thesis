import pandas as pd

data = {
    'Experiment': ['Baseline random', 'Baseline gpy', 'Time agent (argmax)', 'No time agent (argmax)', 'Time agent (sampling)', 'No time agent (sampling)'],
    'n': [50000] * 6,
    'Reward': [1.199314, 2.225365, 2.442427, 2.015795, 2.71864, 2.388971],
    'Reward Error': [0.002588, 0.005858, 0.006517, 0.006751, 0.006716, 0.006351],
    'Length': [18.08691, 18.295829, 26.548903, 20.187176, 21.296037, 18.230957],
    'Length Error': [0.016286, 0.021084, 0.027036, 0.035653, 0.018094, 0.016969],
    'Peak': [0.840579, 0.934053, 0.935286, 0.88822, 0.956154, 0.943636],
    'Peak Error': [0.00052, 0.000407, 0.000423, 0.000543, 0.000324, 0.000362],
    'Log-transformed simple regret': [0.797454, 1.180805, 1.189002, 0.951636, 1.35807, 1.248998],
    'Simple regret': [0.159421, 0.065947, 0.064714, 0.11178, 0.043846, 0.056364]
}

df = pd.DataFrame(data)
df.set_index('Experiment', inplace=True)

# Select the columns you want to include in the table
df_table = df[['Peak', 'Length']]

# Convert the DataFrame to a latex table
latex_table = df_table.to_latex(index=True)

# Print the latex table
print(latex_table)

exit()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Convert index to list for x-axis
experiments = df.index.tolist()

# Define bar position and width
x_pos = np.arange(len(experiments))
width = 0.75

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(12, 6))

# Create bar plot
bars = ax.bar(x_pos, df['Peak'], width=width, yerr=df['Peak Error'], log=True, capsize=7, color='skyblue', edgecolor='grey')

# Set the ticks and labels on the x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(experiments, rotation='vertical')

# Set labels and title
ax.set_ylabel('Peak (log scale)')
ax.set_title('Peak by Experiment')

plt.tight_layout()
plt.show()



# Plotting "Reward" for each experiment
df['Reward'].plot(kind='bar', yerr=df['Reward Error'], capsize=4)
plt.ylabel('Reward')
plt.title('Reward by Experiment')
plt.show()

# Plotting "Length" for each experiment
df['Length'].plot(kind='bar', yerr=df['Length Error'], capsize=4)
plt.ylabel('Length')
plt.title('Length by Experiment')
plt.show()

# Plotting "Peak" for each experiment
df['Peak'].plot(kind='bar', yerr=df['Peak Error'], capsize=4)
plt.ylabel('Peak')
plt.title('Peak by Experiment')
plt.show()

# Plotting "Log-transformed simple regret" for each experiment
df['Log-transformed simple regret'].plot(kind='bar')
plt.ylabel('Log-transformed Simple Regret')
plt.title('Log-transformed Simple Regret by Experiment')
plt.show()

# Plotting "Simple regret" for each experiment
df['Simple regret'].plot(kind='bar')
plt.ylabel('Simple Regret')
plt.title('Simple Regret by Experiment')
plt.show()
