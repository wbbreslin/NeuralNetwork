import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.ticker import StrMethodFormatter

file_path = 'iris_analysis2.csv'
analysis = pd.read_csv(file_path)
print(analysis.head())

# Calculate the correlation coefficient
correlation, _ = pearsonr(analysis['FSO Impact'], analysis['OSE Impact'])

# Create the scatter plot with a dashed black regression line
plt.figure(figsize=(10, 6))

# Plot the regression line
sns.regplot(x="FSO Impact", y="OSE Impact", data=analysis, ci=None, line_kws={"color": "black", "linestyle": "--"})

# Plot the scatter points colored by class using the Viridis color palette
sns.scatterplot(x="FSO Impact", y="OSE Impact", hue="True Class", palette='viridis', data=analysis)

# Set the number of decimal places for x-axis and y-axis labels to 2
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

# Annotate the plot with the correlation coefficient (truncated to 3 decimals)
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
         transform=plt.gca().transAxes,
         ha='left', va='top',
         fontsize=12,
         bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

plt.title("Correlation Between FSO and OSE (N=120)")
# Display the plot
plt.show()
