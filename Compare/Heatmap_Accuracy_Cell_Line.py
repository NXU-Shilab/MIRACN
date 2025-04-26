from keras.saving.saving_api import load_model
from dataset_independent import dataset_make_test, dataset_make_CADD, dataset_make_Expecto, dataset_make_GTEx, dataset_make_Cell_Line

# Load dataset
X, X_info = dataset_make_Cell_Line()

# Load the model
model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")

# Predict functionality probabilities
y_pred_functionality_proba = model.predict(X)[1]

# Define class names
classes = ["GM12878", "GM18507", "HaCaT", "HEK293FT", "HEK293T", "HepG2", "K562"]

# Append predicted probabilities to the X_info DataFrame
for i, class_name in enumerate(classes):
    X_info[class_name] = y_pred_functionality_proba[:, i]

# Save the results to a CSV file
X_info.to_csv("../end_product/heatmap_cell_line.csv", index=False)
