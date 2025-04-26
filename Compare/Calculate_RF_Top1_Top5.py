import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical  # Use Keras one-hot encoding

# Get training and test set data
x_train_temp, y_train_cell_type_temp, _ = dataset_make_train()

# Convert training labels to one-hot encoding form
y_train_cell_type_temp = to_categorical(np.array(y_train_cell_type_temp - 1), 7)

# Get test set data
x_test, y_test_cell_type, _, _, _, _, _ = dataset_make_test()

# Convert y_train_cell_type and y_test_cell_type to single integer labels (for training and evaluation)
y_train_labels = np.argmax(y_train_cell_type_temp, axis=1)
y_test_labels = np.argmax(y_test_cell_type, axis=1)

# Create and train a random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_temp, y_train_labels)

# Save the trained model to a file
model_filename = 'random_forest_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the saved model
loaded_model = joblib.load(model_filename)

# Predict the probability of cell types (output as probabilities for each class)
y_pred_proba = loaded_model.predict_proba(x_test)

# Cell type names
classes = ["GM12878", "GM18507", "HaCaT", "HEK293FT", "HEK293T", "HepG2", "K562"]

# Calculate the Top-1 error rate
y_pred_top1 = np.argmax(y_pred_proba, axis=1)
top1_error = 1 - accuracy_score(y_test_labels, y_pred_top1)
print(f"Top-1 Error: {top1_error:.4f}")

# Calculate the Top-5 error rate
top5_correct = 0
for i in range(len(y_test_labels)):
    # Get the predicted probability values for each sample
    predicted_probs = y_pred_proba[i]
    # Find the indices of the top 5 highest probabilities
    top5_preds = np.argsort(predicted_probs)[-5:]
    # Check if the true label is in the Top-5 predictions
    if y_test_labels[i] in top5_preds:
        top5_correct += 1

# Calculate the Top-5 error rate
top5_error = 1 - (top5_correct / len(y_test_labels))
print(f"Top-5 Error: {top5_error:.4f}")    
