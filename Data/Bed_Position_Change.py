import pandas as pd

# Read the GTEx_use_38_col3.txt file
col3_data = pd.read_csv('../independent_test_GTEx/GTEx-variant-data_38_col3.txt', header=None, sep='\t', names=['chrom_pos'])

# Read the GTEx-variant-data_38.tsv file
variant_data = pd.read_csv('../independent_test_GTEx/GTEx-variant-data_38.tsv', header=None, sep='\t')

# Replace the content of the second column
variant_data.iloc[:, 1] = col3_data['chrom_pos']

# Save to a new file
variant_data.to_csv('GTEx-variant-data_38_replaced.tsv', index=False, header=False, sep='\t')
