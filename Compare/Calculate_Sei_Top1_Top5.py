import numpy as np
from sklearn.metrics import accuracy_score
from dataset import dataset_make_test
# sei_avg is a two-dimensional array storing the probabilities of variants belonging to seven cell lines calculated by the Sei model
# y_test_cell_type is the actual cell line labels in one-hot encoding form
x_test, y_test_cell_type, y_test_compilation, seqclass, label, sei_avg, y_cell_type = dataset_make_test()

# Convert y_test_cell_type to integer labels
y_test_labels = np.argmax(y_test_cell_type, axis=1)
print(sei_avg)
# Calculate the Top-1 error rate
y_pred_top1 = np.argmax(sei_avg, axis=1)
top1_error = 1 - accuracy_score(y_test_labels, y_pred_top1)
print(f"Top-1 Error: {top1_error:.4f}")
print(y_pred_top1)
# Calculate the Top-5 error rate
sei_avg = sei_avg.values
top5_correct = 0
for i in range(len(y_test_labels)):
    # Get the predicted probability values for each sample
    predicted_probs = sei_avg[i]
    # Find the indices of the top 5 highest probabilities
    top5_preds = np.argsort(predicted_probs)[-5:]
    # Check if the true label is in the Top-5 predictions
    if y_test_labels[i] in top5_preds:
        top5_correct += 1

# Calculate the Top-5 error rate
top5_error = 1 - (top5_correct / len(y_test_labels))
print(f"Top-5 Error: {top5_error:.4f}")    
