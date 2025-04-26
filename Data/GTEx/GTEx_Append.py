import pandas as pd

# Read the label.txt file
label_df = pd.read_csv('label.txt', sep='\t', header=None)

# Read the GTEx-variant-data_tabix.csv file
gtex_df = pd.read_csv('GTEx-variant-data_tabix.csv', header=None)

# Check if the number of rows in both is the same
print(len(gtex_df))
print(len(label_df))
if len(gtex_df) != len(label_df):
    raise ValueError("The number of rows in GTEx-variant-data_tabix.csv and label.txt files is not the same")

# Set the column names of label_df to prevent conflicts with gtex_df column names
label_df.columns = [f'label_{i+1}' for i in range(label_df.shape[1])]

# Append the columns in label_df to the last three columns of gtex_df
result_df = pd.concat([gtex_df, label_df], axis=1)

# Save the new CSV file
result_df.to_csv('GTEx-variant-data_tabix_label.csv', index=False)
