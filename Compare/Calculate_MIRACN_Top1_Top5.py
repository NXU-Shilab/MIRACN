import numpy as np
from keras.saving.saving_api import load_model
from dataset import dataset_make_test
from sklearn.metrics import accuracy_score

# Get the test set data
x_test, y_test_cell_type, _, _, _, _, _ = dataset_make_test()

# Load the model
model = load_model("/mnt/data0/users/lizy/pycharm_project/model_1.h5")

# Predict the probability of cell types
y_pred_functionality_proba = model.predict(x_test)[1]
print(f"Type of y_pred_functionality_proba: {type(y_pred_functionality_proba)}")
# Cell type names
classes = ["GM12878", "GM18507", "HaCaT", "HEK293FT", "HEK293T", "HepG2", "K562"]

# Get the actual cell type labels (convert to integer label form)
y_true_labels = np.argmax(y_test_cell_type, axis=1)

# Calculate the Top-1 error rate
y_pred_top1 = np.argmax(y_pred_functionality_proba, axis=1)
top1_error = 1 - accuracy_score(y_true_labels, y_pred_top1)
print(f"Top-1 Error: {top1_error:.4f}")

# Calculate the Top-5 error rate
top5_correct = 0
for i in range(len(y_true_labels)):
    # Extract the predicted probability values for each sample
    predicted_probs = y_pred_functionality_proba[i]
    # Find the indices of the top 5 highest probabilities
    top5_preds = np.argsort(predicted_probs)[-5:]
    # Check if the true label is in the Top-5 predictions
    if y_true_labels[i] in top5_preds:
        top5_correct += 1

# Calculate the Top-5 error rate
top5_error = 1 - (top5_correct / len(y_true_labels))
print(f"Top-5 Error: {top5_error:.4f}")
