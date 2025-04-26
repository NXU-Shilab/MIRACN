import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from keras.saving.saving_api import load_model
from dataset_ind_fig import dataset_make_test, dataset_make_DVAR
X, label, seqclass = dataset_make_test()
DVAR, label_DVAR = dataset_make_DVAR()

# Calculate the optimal threshold
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")
y_pred_compilation_proba = model.predict(X)[0]

# Calculate the optimal threshold
threshold_test = Find_Optimal_Cutoff(label, y_pred_compilation_proba)
threshold_Sei = Find_Optimal_Cutoff(label, seqclass)
threshold_DVAR = Find_Optimal_Cutoff(label_DVAR, DVAR)

# threshold_test = pd.DataFrame(threshold_test)
print('threshold_test_{}', threshold_test)
print('threshold_test_{}', threshold_Sei)
print('threshold_test_{}', threshold_DVAR)
