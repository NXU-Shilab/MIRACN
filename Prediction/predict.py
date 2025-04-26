import argparse
import pandas as pd
import numpy as np
from keras.models import load_model

# Create an argument parser
parser = argparse.ArgumentParser(description='Load data, make predictions, and save results.')
parser.add_argument('file_name', type=str, help='Path to the data file')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
X_test = data.iloc[:, 5:]

# Load model
model_path = 'MIRACN.h5'
loaded_model = load_model(model_path)

# Make predictions using the loaded model
X_test_fold = np.array(X_test)

y_Functionality_Proba = loaded_model.predict(X_test_fold)[0]
y_Cell_Line_Proba = loaded_model.predict(X_test_fold)[1]

output_data = data.iloc[:, :5]

output_data['Functionality_Probability'] = y_Functionality_Proba

classes = ["GM12878", "GM18507", "HaCaT", "HEK293FT", "HEK293T", "HepG2", "K562"]

for i, class_name in enumerate(classes):
    output_data[class_name] = y_Cell_Line_Proba[:, i]

output_file_name = args.file_name.replace('.csv', '_predictions.csv')
output_data.to_csv(output_file_name, index=False, sep='\t')
print(f'Predicted probabilities saved to {output_file_name}')
