import os
import seaborn as sns
from keras.saving.saving_api import load_model
from matplotlib.backends.backend_pdf import PdfPages
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from scipy.stats import uniform, randint
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn
from keras.layers import Conv1D, MaxPooling1D
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
from train_nois import train_5f
from dataset import dataset_make_test, dataset_make_CADD, dataset_make_Expecto, dataset_make_DVAR
from model import create_model

# Load dataset
x_test, y_test_cell_type, y_test_compilation, seqclass, label, sei_avg, y_cell_type = dataset_make_test()

# Load the model
model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")

# Predict cell type probabilities
y_pred_cell_type_proba_a = model.predict(x_test)[1]
arr = np.array(y_pred_cell_type_proba_a)

# Initialize a list to store column accuracies
column_accuracies_1 = []
y_cell_type_sei_proba_a = np.array([np.where(x == np.max(x), 1, 0) for x in arr])

# Compute AUC for each column
for col in range(arr.shape[1]):
    y_cell_type_proba_col = y_cell_type_sei_proba_a[:, col]
    y_cell_type_col = y_test_cell_type[:, col]
    auc_col = roc_auc_score(y_cell_type_col, y_cell_type_proba_col)
    column_accuracies_1.append(auc_col)

# Calculate the average AUC
total_auc_1 = np.mean(column_accuracies_1)
column_accuracies_1.append(total_auc_1)
column_accuracies_1 = np.array(column_accuracies_1)
print("Column Accuracies_1:", column_accuracies_1)

# Convert Sei predictions to binary format
arr_sei = np.array(sei_avg)
column_accuracies = []
y_cell_type_sei_proba = np.array([np.where(x == np.max(x), 1, 0) for x in arr_sei])

# Compute AUC for each column
for col in range(arr_sei.shape[1]):
    y_cell_type_sei_proba_col = y_cell_type_sei_proba[:, col]
    y_cell_type_col = y_test_cell_type[:, col]
    column_auc = roc_auc_score(y_cell_type_col, y_cell_type_sei_proba_col)
    column_accuracies.append(column_auc)

# Calculate the average AUC
total_auc_2 = np.mean(column_accuracies)
column_accuracies.append(total_auc_2)
column_accuracies = np.array(column_accuracies)
print("Column Accuracies:", column_accuracies)

# Combine accuracies into a 2D array
combined_accuracies = np.vstack((column_accuracies, column_accuracies_1))

# Plot heatmap
plt.figure(figsize=(6, 6))  # Set figure size to square
vegetables = ["Sei", "MIRACN"]
farmers = ["GM12878", "GM18507", "HaCaT", "HEK293FT", "HEK293T", "HepG2", "K562", "Total"]

# Create heatmap
fig, ax = plt.subplots()
sns.heatmap(combined_accuracies, annot=True, cmap='viridis', xticklabels=False, yticklabels=False, fmt='.3f')

# Set x and y axis tick positions
x_ticks = np.arange(len(farmers))
y_ticks = np.arange(len(vegetables))

# Set x-axis tick labels
ax.set_xticks(x_ticks + 0.5)
ax.set_xticklabels(farmers, rotation=45, ha="right", va="center", rotation_mode="anchor")

# Set y-axis tick labels
ax.set_yticks(y_ticks + 0.5)
ax.set_yticklabels(vegetables, va="center")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('end_product/heatmap_auc.pdf')
