import pandas as pd

# Read the first file (GTEx_use_label_deleted_38.tsv)
file1 = '../independent_test_GTEx/GTEx-variant-data_38_replaced_with_header.tsv'
df1 = pd.read_csv(file1, sep='\t')

# Create a join key for the first file
df1['key'] = df1.iloc[:, 0].str.replace('chr', '') + '_' + df1.iloc[:, 1].astype(str)

# Read the second file (result_GTEx_38.csv)
file2 = '../end_product/result_GTEx_38_rm_train.csv'
df2 = pd.read_csv(file2, sep=',')

# Remove the 'chr' prefix from the first column and create a join key
df2['key'] = df2.iloc[:, 0].str.replace('chr', '') + '_' + df2.iloc[:, 1].astype(str)

# Merge the two DataFrames
merged_df = pd.merge(df1, df2, on='key')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('../end_product/merged_result_GTEx_38_rm_train.csv', index=False)
