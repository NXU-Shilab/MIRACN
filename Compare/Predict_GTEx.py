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
from dataset_independent import dataset_make_test, dataset_make_CADD, dataset_make_Expecto, dataset_make_GTEx
from model import create_model


X ,X_info= dataset_make_GTEx();


model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")
y_pred_compilation_proba = model.predict(X)[0]
X_info["Proba"] = y_pred_compilation_proba
# combined_df = pd.concat([y_pred_compilation_proba, X], axis=1)
X_info.to_csv("end_product/result_GTEx_38.csv", index=False)
