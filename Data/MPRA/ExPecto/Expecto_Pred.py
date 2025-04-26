import pandas as pd

# Read the CSV file
df = pd.read_csv('my_ExPecto/output_l_1.csv')

# Get the data from the 12th column to the last column
data = df.iloc[:, 12:]

# Check if any absolute value is greater than 0.3
df['pred'] = (data.abs() > 0.3).any(axis=1)

# Convert True and False to 1 and 0
df['pred'] = df['pred'].astype(int)

# Save the result to a new CSV file
df.to_csv('expecto_pred.csv', index=False)
