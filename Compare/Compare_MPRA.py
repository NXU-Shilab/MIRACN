import os
import pandas as pd
import seaborn as sns
from keras.saving.saving_api import load_model
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import regularizers
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, auc, accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from dataset import dataset_make_test, dataset_make_CADD, dataset_make_Expecto, dataset_make_DVAR


x_test, y_test_cell_type, y_test_compilation, seqclass, label, sei_avg, y_cell_type = dataset_make_test()
label_expecto, pre_expecto = dataset_make_Expecto();
cadd, label_cadd = dataset_make_CADD()
DVAR, label_DVAR = dataset_make_DVAR()
seqclass_aucs = []
cadd_auc = []
cell_type_accuracies_sei = []
DVAR_auc = []
# fold_results, cell_type_accuracies, compilation_aucs = train_5f()
# Calculate ROC AUC
y_test_compilation_fold = np.asarray(y_test_compilation)
seqclass = np.array(seqclass)
label = np.array(label)
cadd = np.array(cadd)
label_cadd = np.array(label_cadd)
pre_expecto = np.array(pre_expecto)
label_expecto = np.array(label_expecto)
DVAR = np.array(DVAR)
label_DVAR = np.array(label_DVAR)

seqclass_auc = roc_auc_score(label, seqclass)
cadd_auc = roc_auc_score(label_cadd, cadd)
expecto_auc = roc_auc_score(label_expecto, pre_expecto)
DVAR_auc = roc_auc_score(label_DVAR, DVAR)
fpr_seqclass, tpr_seqclass, thresholds1 = roc_curve(label, seqclass)
fpr_cadd, tpr_cadd, thresholds2 = roc_curve(label_cadd, cadd)
fpr_expecto, tpr_expecto, thresholds3 = roc_curve(label_expecto, pre_expecto)
fpr_DVAR, tpr_DVAR, thresholds4 = roc_curve(label_DVAR, DVAR)

roc_auc_seqclass = auc(fpr_seqclass, tpr_seqclass)
roc_auc_cadd = auc(fpr_cadd, tpr_cadd)
roc_auc_expecto = auc(fpr_expecto, tpr_expecto)
roc_auc_DVAR = auc(fpr_DVAR, tpr_DVAR)

model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")
y_pred_compilation_proba = model.predict(x_test)[0]
fpr_compilation, tpr_compilation, thresholds = roc_curve(y_test_compilation_fold, y_pred_compilation_proba)
roc_auc_compilation = auc(fpr_compilation, tpr_compilation)

# Calculate Precision-Recall AUC
precision_compilation, recall_compilation, _ = precision_recall_curve(y_test_compilation_fold, y_pred_compilation_proba)
auprc_compilation = auc(recall_compilation, precision_compilation)
precision_seqclass, recall_seqclass, _ = precision_recall_curve(label, seqclass)
auprc_seqclass = auc(recall_seqclass, precision_seqclass)
precision_cadd, recall_cadd, _ = precision_recall_curve(label_cadd, cadd)
auprc_cadd = auc(recall_cadd, precision_cadd)
precision_expecto, recall_expecto, _ = precision_recall_curve(label_expecto, pre_expecto)
auprc_expecto = auc(recall_expecto, precision_expecto)
precision_DVAR, recall_DVAR, _ = precision_recall_curve(label_DVAR, DVAR)
auprc_DVAR = auc(recall_DVAR, precision_DVAR)

# Round AUPRC and AUROC to three decimal places
results = {
    'Model': ['MIRACN', 'Sei', 'CADD', 'Expecto', 'DVAR'],
    'AUROC': [round(roc_auc_compilation, 3), round(roc_auc_seqclass, 3), round(roc_auc_cadd, 3), round(roc_auc_expecto, 3), round(roc_auc_DVAR, 3)],
    'AUPRC': [round(auprc_compilation, 3), round(auprc_seqclass, 3), round(auprc_cadd, 3), round(auprc_expecto, 3), round(auprc_DVAR, 3)]
}

# Create a DataFrame and save it as a CSV file
df_results = pd.DataFrame(results)
df_results.to_csv('end_product/AUC_AUPRC_results.csv', index=False)

# Print the result table
print(df_results)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
for name, AUC in zip(results['Model'], results['AUROC']):
    if name == 'MIRACN':
        plt.plot(fpr_compilation, tpr_compilation, label=f'{name} (AUC = {AUC:.3f})')
    elif name == 'Sei':
        plt.plot(fpr_seqclass, tpr_seqclass, label=f'{name} (AUC = {AUC:.3f})')
    elif name == 'CADD':
        plt.plot(fpr_cadd, tpr_cadd, label=f'{name} (AUC = {AUC:.3f})')
    elif name == 'Expecto':
        plt.plot(fpr_expecto, tpr_expecto, label=f'{name} (AUC = {AUC:.3f})')
    elif name == 'DVAR':
        plt.plot(fpr_DVAR, tpr_DVAR, label=f'{name} (AUC = {AUC:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('end_product/AUROC_Test.pdf')
plt.show()
plt.close()

# Plot the Precision-Recall curve
plt.figure(figsize=(6, 6))
for name, auprc in zip(results['Model'], results['AUPRC']):
    if name == 'MIRACN':
        plt.plot(recall_compilation, precision_compilation, label=f'{name} (AUPRC = {auprc:.3f})')
    elif name == 'Sei':
        plt.plot(recall_seqclass, precision_seqclass, label=f'{name} (AUPRC = {auprc:.3f})')
    elif name == 'CADD':
        plt.plot(recall_cadd, precision_cadd, label=f'{name} (AUPRC = {auprc:.3f})')
    elif name == 'Expecto':
        plt.plot(recall_expecto, precision_expecto, label=f'{name} (AUPRC = {auprc:.3f})')
    elif name == 'DVAR':
        plt.plot(recall_DVAR, precision_DVAR, label=f'{name} (AUPRC = {auprc:.3f})')

plt.legend(loc='lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.savefig('end_product/AUPRC_Test.pdf')
plt.show()
plt.close()
