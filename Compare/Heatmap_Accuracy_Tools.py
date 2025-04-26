import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('heatmap_tools.tsv', sep='\t', header=None)

# Set the index and column names
column_names = ['chrom_x', 'pos_x', 'MIRACN', 'Sei', 'CADD', 'DVAR', 'ExPecto']
data.columns = column_names

# Add a new index column
data['Position'] = data['chrom_x'].astype(str) + ':' + data['pos_x'].astype(str)
data.set_index('Position', inplace=True)

# Threshold definition
thresholds = {
    'MIRACN': 0.1580081582069397,
    'CADD': 10,
    'Sei': 0.0778954124895996,
    'DVAR': 0.6336,
    'ExPecto': 0.3
}

# Convert the probability columns to numeric type
for col in ['MIRACN', 'Sei', 'CADD', 'DVAR', 'ExPecto']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.sort_values(by='pos', inplace=True)
# Normalization processing
for tool, threshold in thresholds.items():
    # Check if there are non-numeric data converted to NaN
    data[tool] = (data[tool] - threshold) / threshold

# Print the normalized data
print("Normalized data:")
print(data[['MIRACN', 'Sei', 'CADD', 'DVAR', 'ExPecto']])

# Draw a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data[['MIRACN', 'Sei', 'CADD', 'DVAR', 'ExPecto']], annot=True, fmt=".3f", cmap="YlGnBu")

# Set the title and axis labels
plt.xlabel('Tools')
plt.ylabel('ID')

# Save as PDF
plt.savefig('heatmap_tools.pdf', format='pdf')
