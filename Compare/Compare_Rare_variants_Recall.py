from keras.saving.saving_api import load_model
from sklearn.metrics import recall_score, confusion_matrix

from dataset_ind_fig import dataset_make_test, dataset_make_Expecto, dataset_make_CADD,dataset_make_DVAR

X, y_test_functionality, seqclass = dataset_make_test()
label_expecto, pre_expecto = dataset_make_Expecto();
PHRED, label_CADD = dataset_make_CADD()
probability_DVAR, label_DVAR = dataset_make_DVAR();
model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")
y_pred_functionality_proba = model.predict(X)[0]
threshold =  0.1580081582069397 
threshold_CADD = 10
threshold_CADD_1 = 6.4
threshold_Sei = 0.0778954124895996
threshold_DVAR = 0.6336
y_pred_functionality = (y_pred_functionality_proba >= threshold).astype(int)
y_CADD_10 =  (PHRED >= threshold_CADD).astype(int)
y_CADD_6 =  (PHRED >= threshold_CADD_1).astype(int)
y_SEI = (seqclass >= threshold_Sei).astype(int)
y_DVAR = (probability_DVAR >= threshold_DVAR).astype(int)

cm = confusion_matrix(y_test_functionality, y_pred_functionality)
cm_ex = confusion_matrix(label_expecto,pre_expecto)
cm_CADD = confusion_matrix(label_CADD,y_CADD_10)
cm_CADD_6 = confusion_matrix(label_CADD,y_CADD_6)
cm_SEI = confusion_matrix(y_test_functionality,y_SEI)
cm_DVAR = confusion_matrix(label_DVAR,y_DVAR)

true_positives = cm[1, 1]
true_positives_1 = cm_ex[1, 1]
true_positives_2 = cm_CADD[1,1]
true_positives_3 = cm_CADD_6[1,1]
true_positives_4 = cm_SEI[1, 1]
true_positives_5 = cm_DVAR[1, 1]
print(f"Number of True Positives (MIRACN): {true_positives}")
print(f"Number of True Positives (Expecto): {true_positives_1}")
print(f"Number of True Positives (CADD_10): {true_positives_2}")
print(f"Number of True Positives (CADD_6): {true_positives_3}")
print(f"Number of True Positives (SEI): {true_positives_4}")
print(f"Number of True Positives (DVAR): {true_positives_5}")
