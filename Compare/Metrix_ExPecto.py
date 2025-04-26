# Calculate TP, FP, TN, FN

from sklearn.metrics import precision_recall_curve, roc_curve, auc

import pandas as pd

import numpy as np

def compute_confusion_matrix(precited, expected):
    # Classify the results. XOR makes the correctly judged results 0 and the incorrectly judged results 1.
    part = precited ^ expected
    # Count the classification results. pcount[0] is the number of 0s, pcount[1] is the number of 1s.
    pcount = np.bincount(part)
    # Convert the calculation result of TP to a list.
    tp_list = list(precited & expected)
    # Convert the calculation result of FP to a list.
    fp_list = list(precited & ~expected)
    # Count the number of TPs.
    TP = tp_list.count(1)
    # Count the number of FPs.
    FP = fp_list.count(1)
    # Count the number of TNs.
    TN = pcount[0] - TP
    # Count the number of FNs.
    FN = pcount[1] - FP
    return TP, FP, TN, FN

file = pd.read_csv('expecto_pred.csv')
label = file['5']
y_pred = file['pred']
# Save the metrics after each fold of cross - validation.
accuracies = []
specificities = []
precisions = []
recalls = []
f1_scores = []
aurocs = []
auprcs = []

TP, FP, TN, FN = compute_confusion_matrix(label, y_pred)

# Calculate accuracy
accuracy = (TP + TN) / (TP + FP + TN + FN)
accuracies.append(accuracy)

# Calculate specificity
specificity = TN / (TN + FP)
specificities.append(specificity)

# Calculate precision
precision = TP / (TP + FP)
precisions.append(precision)

# Calculate recall
recall = TP / (TP + FN)
recalls.append(recall)

# Calculate f1 - score
f1_score = (2 * precision * recall) / (precision + recall)
f1_scores.append(f1_score)

# Calculate AUC
fpr, tpr, threshold = roc_curve(label, y_pred)
auroc = auc(fpr, tpr)
aurocs.append(auroc)

# Calculate AUPRC
precision_, recall_, _ = precision_recall_curve(label, y_pred)
auprc = auc(recall_, precision_)
auprcs.append(auprc)

print('TP: {}'.format(TP))
print('FP: {}'.format(FP))
print('TN: {}'.format(TN))
print('FN: {}'.format(FN))
print("Accuracy: {:.3f}".format(accuracy))
print("specificity: {:.3f}".format(specificity))
print("precision: {:.3f}".format(precision))
print("recall: {:.3f}".format(recall))
print("f1_score: {:.3f}".format(f1_score))
print("auroc: {:.3f}".format(auroc))
print("auprc: {:.3f}".format(auprc))    
