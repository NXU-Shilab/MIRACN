import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


# Get the training set and test set
def dataset_make_train():
    file1 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_1.csv')
    file2 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_0.csv')

    file1['label'] = 1
    file2['label'] = 0

    data = pd.concat([file1, file2], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop(["seqclass_max_absdiff", "index", "chrom", "pos", "name", "ref", "alt", "strand", "ref_match",
                   "contains_unk", "cell_line", "label", 'GM12878_avg', 'GM18507_avg', 'HaCaT_avg', 'HEK293FT_avg',
                   'HEK293T_avg', 'HepG2_avg', 'K562_avg'], axis=1)
    y_cell_type = data['cell_line']
    y_functionality = data['label']

    x_train_temp, x_test, y_train_cell_type_temp, y_test_cell_type, y_train_functionality_temp, y_test_functionality = train_test_split(
        X, y_cell_type, y_functionality, test_size=0.1, random_state=42)

    return x_train_temp, y_train_cell_type_temp, y_train_functionality_temp


def dataset_make_test():
    file1 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_1.csv')
    file2 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_0.csv')

    file1['label'] = 1
    file2['label'] = 0

    data = pd.concat([file1, file2], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop(["seqclass_max_absdiff", "index", "chrom", "pos", "name", "ref", "alt", "strand", "ref_match",
                   "contains_unk", "cell_line", "label", 'GM12878_avg', 'GM18507_avg', 'HaCaT_avg', 'HEK293FT_avg',
                   'HEK293T_avg', 'HepG2_avg', 'K562_avg'], axis=1)
    y_cell_type = data['cell_line']
    y_functionality = data['label']

    x_train_temp, x_test, y_train_cell_type_temp, y_test_cell_type, y_train_functionality_temp, y_test_functionality = train_test_split(
        X, y_cell_type, y_functionality, test_size=0.1, random_state=42)

    return x_test, y_test_cell_type, y_test_functionality


# Load the data
x_train_temp, y_train_cell_type_temp, y_train_functionality_temp = dataset_make_train()
x_test, y_test_cell_type, y_test_functionality = dataset_make_test()

# Create a Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(x_train_temp, y_train_functionality_temp)

# Split according to the cell line
cell_lines = np.unique(y_test_cell_type)
auc_scores = []
auprc_scores = []
cell_lines_list = []

# Iterate through each cell line for individual prediction
for cell_line in cell_lines:
    # Get the test set for this cell line
    idx = (y_test_cell_type == cell_line)
    x_test_cell_line = x_test[idx]
    y_test_functionality_cell_line = y_test_functionality[idx]

    # Make predictions and calculate probabilities
    y_pred_prob = rf.predict_proba(x_test_cell_line)[:, 1]  # Get the probability of the positive class

    # Calculate AUC
    auc_score = roc_auc_score(y_test_functionality_cell_line, y_pred_prob)
    auc_scores.append(auc_score)

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_test_functionality_cell_line, y_pred_prob)
    auprc_score = auc(recall, precision)
    auprc_scores.append(auprc_score)

    # Save the corresponding cell line name
    cell_lines_list.append(f"Cell Line {cell_line}")

# Calculate the AUC and AUPRC for the entire test set
y_pred_prob_test = rf.predict_proba(x_test)[:, 1]
auc_test = roc_auc_score(y_test_functionality, y_pred_prob_test)

precision_test, recall_test, _ = precision_recall_curve(y_test_functionality, y_pred_prob_test)
auprc_test = auc(recall_test, precision_test)

# Add the total AUC and AUPRC to the list
cell_lines_list.append("Overall Test Set")
auc_scores.append(auc_test)
auprc_scores.append(auprc_test)

# Create a DataFrame and save it to a CSV
results_df = pd.DataFrame({
    'Cell Line': cell_lines_list,
    'AUC': auc_scores,
    'AUPRC': auprc_scores
})

# Save as a CSV file
results_df.to_csv("auc_auprc_results.csv", index=False)

# Print the output
print(results_df)
