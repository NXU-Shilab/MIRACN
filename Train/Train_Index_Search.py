import os
import keras.optimizers
import seaborn as sns
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

from dataset import dataset_make_train
from model import create_model

warnings.filterwarnings('ignore')

# % matplotlib
# inline


def train_5f():
    out_dir = '/mnt/data0/users/lizy/pycharm_project'

    # Load the data
    num_classes = 7
    print("Loading data")
    x_train_temp, y_train_cell_type_temp, y_train_compilation_temp = dataset_make_train()
    print("Load over")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    compilation_aucs = []
    cell_type_accuracies = []
    fold_results = []

    # Define the base model

    # Define the random search parameters
    param_grid = {
        'optimizer': ['adam','sgd', 'rmsprop'],
        'dropout_rate': [0.1, 0.2, 0.3],
        'neurons': [32, 64, 128],
        'kernel_size': [3, 4, 5]
    }

    # Create Randomized Search
    param_combinations = ParameterGrid(param_grid)

    for params_idx, params in enumerate(param_combinations, start=1):
        optimizer = params['optimizer']
        dropout_rate = params['dropout_rate']
        neurons = params['neurons']
        kernel_size = params['kernel_size']
        print(f'\n--- Parameter Combination {params_idx} ---')
        print(f'Optimizer: {optimizer}, Dropout Rate: {dropout_rate}, Neurons: {neurons}, Kernel Size: {kernel_size}')

        # print('Start five-fold cross-validation')
        i = 1
        for fold, (train_index, val_index) in enumerate(kf.split(x_train_temp, y_train_compilation_temp)):
            x_train_fold, x_val_fold = x_train_temp.iloc[train_index], x_train_temp.iloc[val_index]
            # print(x_train_fold.shape,x_val_fold.shape)
            y_train_cell_type_fold, y_val_cell_type_fold = y_train_cell_type_temp.iloc[train_index], \
                y_train_cell_type_temp.iloc[val_index]
            # print(y_train_cell_type_fold.shape,y_val_cell_type_fold.shape)
            y_train_compilation_fold, y_val_compilation_fold = y_train_compilation_temp.iloc[train_index], \
                y_train_compilation_temp.iloc[val_index]
            # print(y_train_compilation_fold.shape, y_val_compilation_fold.shape)
            # Prepare the training and validation sets
            y_train_cell_type_fold = to_categorical(np.array(y_train_cell_type_fold - 1), num_classes)
            y_val_cell_type_fold = to_categorical(np.array(y_val_cell_type_fold - 1), num_classes)
            x_train_fold = np.array(x_train_fold)
            x_val_fold = np.array(x_val_fold)
            y_train_compilation_fold = np.array(y_train_compilation_fold)
            y_val_compilation_fold = np.array(y_val_compilation_fold)
            # print(x_train_fold.shape,y_train_compilation_temp.shape,y_train_cell_type_temp.shape)

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (x_train_fold, (y_train_compilation_fold, y_train_cell_type_fold)))
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val_fold, (y_val_compilation_fold, y_val_cell_type_fold)))
            train_dataset = train_dataset.shuffle(2000, reshuffle_each_iteration=True).batch(128).prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

            # Train the model

            model = create_model(dropout_rate, neurons, kernel_size, optimizer)
            callbacks = [
                EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=1,
                              restore_best_weights=True)
            ]

            print(f"\nStart training - Fold {i}")
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=100,  # Can be adjusted appropriately
                verbose=1,
                callbacks=callbacks

            )
            print("Training end")
            acc = history.history['compilation_accuracy']
            val_acc = history.history['val_compilation_accuracy']

            # Plotting accuracy
            epochs = range(len(acc))

            # Extracting loss values
            loss = history.history['compilation_loss']
            val_loss = history.history['val_compilation_loss']

            acc = history.history['cell_type_accuracy']
            val_acc = history.history['val_cell_type_accuracy']

            epochs = range(len(acc))

            loss = history.history['cell_type_loss']
            val_loss = history.history['val_cell_type_loss']

            # Calculate evaluation metrics for the binary classification task (compilation)
            y_pred_compilation_proba = model.predict(x_val_fold)[0]
            compilation_auc = roc_auc_score(y_val_compilation_fold, y_pred_compilation_proba)
            compilation_aucs.append(compilation_auc)

            y_pred_cell_type_proba = model.predict(x_val_fold)[1]

            arr = np.array(y_pred_cell_type_proba)

            # Convert each dimension to contain only 0 and 1, and only one 1
            y_pred_cell_type_proba = np.array([np.where(x == np.max(x), 1, 0) for x in arr])
            cell_type_accuracy = accuracy_score(y_val_cell_type_fold, y_pred_cell_type_proba)

            # The weight is the frequency of the class in the dataset

            cell_type_accuracies.append(cell_type_accuracy)

            fold_result = {'optimizer': optimizer, 'dropout_rate': dropout_rate, 'neurons': neurons,
                           'kernel_size': kernel_size, 'compilation_auc': compilation_auc,
                           'cell_type_accuracy': cell_type_accuracy}
            fold_results.append(fold_result)
            i = i + 1

    return fold_results, cell_type_accuracies, compilation_aucs
