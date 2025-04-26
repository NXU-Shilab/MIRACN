import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score

# Read the CSV file
df = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/result_GTEx_38_test.csv')
df2 = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/independent_test_GTEx/output_modified/merged_result_GTEx_38_All.csv')

# The sixth column is probability, and the eleventh, twelfth, and thirteenth columns are labels
probability = df.iloc[:, 5]
label1 = df2.iloc[:, 18]
label2 = df2.iloc[:, 19]
label3 = df2.iloc[:, 20]


# predicted_label = probability.round()
threshold = 0.25217345356941223

# Convert the predicted probability to binary classification
predicted_label = (probability >= threshold).astype(int)
# Calculate the accuracy
accuracy1 = accuracy_score(label1, predicted_label)
accuracy2 = accuracy_score(label2, predicted_label)
accuracy3 = accuracy_score(label3, predicted_label)
threshold = 0.25217345356941223

# Calculate the number of true positives
true_positives = np.sum((predicted_label == 1) & (label1 == 1))
p = np.sum(label1 == 1)

print(p)
print("True Positives (TP):", true_positives)

print(f'Accuracy with label1: {accuracy1:.2f}')
print(f'Accuracy with label2: {accuracy2:.2f}')
print(f'Accuracy with label3: {accuracy3:.2f}')
