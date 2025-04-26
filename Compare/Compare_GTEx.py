import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/end_product/merged_result_GTEx_38.csv')

# Define thresholds for each tool
thresholds = {
    'CADD': 10,
    'CADD_6.4': 6.4,
    'GWAVA-Region': 0.5,
    'GWAVA-TSS': 0.5,
    'GWAVA-Unmatch': 0.5,
    'DeepSEA': 0.05,
    'DeepSEA-0.01': 0.01,
    'LINSIGHT': 0.08,
    'Eigen-phred': 5,
    'Eigen-PC-phred': 9,
    'MIRACN': 0.25217345356941223
}

# Create classification columns
df['CADD_10'] = df.iloc[:, 4] > thresholds['CADD']
df['CADD_6.4'] = df.iloc[:, 4] > thresholds['CADD_6.4']
df['GWAVA-Region'] = df.iloc[:, 5] > thresholds['GWAVA-Region']
df['GWAVA-TSS'] = df.iloc[:, 6] > thresholds['GWAVA-TSS']
df['GWAVA-Unmatch'] = df.iloc[:, 7] > thresholds['GWAVA-Unmatch']
df['DeepSEA_0.05'] = df.iloc[:, 8] < thresholds['DeepSEA']
df['DeepSEA_0.01'] = df.iloc[:, 8] < thresholds['DeepSEA-0.01']
df['LINSIGHT'] = df.iloc[:, 9] > thresholds['LINSIGHT']
df['Eigen-phred'] = df.iloc[:, 11] > thresholds['Eigen-phred']
df['Eigen-PC-phred'] = df.iloc[:, 13] > thresholds['Eigen-PC-phred']
df['MIRACN'] = df.iloc[:, 37] > thresholds['MIRACN']

# Initialize results list
results = []

# Calculate recall, AUC, ACC, TP, and TPR
for label_column in [18, 19, 20]:
    label = df.iloc[:, label_column]
    if label_column == 18:
        label_name = '550 eSNVs with largest effect size'
    elif label_column == 19:
        label_name = '425 eSNVs most likely causal'
    elif label_column == 20:
        label_name = '304 eSNVs with best p-value'

    for tool in ['CADD_10', 'CADD_6.4', 'GWAVA-Region', 'GWAVA-TSS', 'GWAVA-Unmatch', 'DeepSEA_0.05', 'DeepSEA_0.01', 'LINSIGHT', 'Eigen-phred', 'Eigen-PC-phred', 'MIRACN']:
        recall = recall_score(label, df[tool])
        auc = roc_auc_score(label, df[tool])
        acc = accuracy_score(label, df[tool])
        tp = ((df[tool] == True) & (label == 1)).sum()
        tpr = tp / label.sum()
        results.append((tool, label_name, recall, auc, acc, tp, tpr))

# Create results DataFrame
results_df = pd.DataFrame(results, columns=['Tool', 'Label', 'Recall', 'AUC', 'ACC', 'TP', 'TPR'])

# Save results to CSV file
results_df.to_csv('../end_product/evaluation_results.csv', index=False)

# Set color palette
palette_tpr = sns.color_palette("Set2", 3)
palette_auc = sns.color_palette("Set1", 3)

# Plot TPR bar chart
plt.figure(figsize=(14, 5))
bar_plot = sns.barplot(data=results_df, x='Tool', y='TPR', hue='Label', dodge=True, palette=palette_tpr)
for bar in bar_plot.patches:
    bar.set_width(0.2)
plt.ylabel('Percent')
plt.xlabel('Tool')
plt.legend(title='', loc="upper left")
plt.tight_layout()
plt.savefig('../end_product/GTEx_FIG_Percent.pdf')
plt.show()

# Plot AUC bar chart
plt.figure(figsize=(14, 5))
bar_plot = sns.barplot(data=results_df, x='Tool', y='AUC', hue='Label', dodge=True, palette=palette_auc)
for bar in bar_plot.patches:
    bar.set_width(0.2)
plt.ylabel('AUC')
plt.xlabel('Tool')
plt.ylim(0.4, 0.6)
plt.yticks([i/40 for i in range(16, 25, 1)])  # 0.4 to 0.6 with step 0.025
plt.legend(title='', loc="upper left")
plt.tight_layout()
plt.savefig('../end_product/GTEx_FIG_Auc.pdf')
plt.show()
