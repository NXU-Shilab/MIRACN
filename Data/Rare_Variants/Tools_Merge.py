import pandas as pd

# Read the files
file1 = pd.read_csv('../independent_test/Sei_test_ind/result_SR.csv', sep=',')
file2 = pd.read_csv('../independent_test/CADD/CADD_no_anno.tsv', sep='\t')
file3 = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/independent_test/DVAR/DVAR_sr_hg38.tsv', sep='\t')
file4 = pd.read_csv('../independent_test/Expecto/Expecto_SR_proba.csv', sep=',')

# Create a join key for each file
file1['key'] = file1.iloc[:, 0].str.replace('chr', '') + '_' + file1.iloc[:, 1].astype(str)
file2['key'] = file2.iloc[:, 0].str.replace('chr', '') + '_' + file2.iloc[:, 1].astype(str)
file3['key'] = file3.iloc[:, 0].str.replace('chr', '') + '_' + file3.iloc[:, 1].astype(str)
file4['key'] = file4.iloc[:, 0].str.replace('chr', '') + '_' + file4.iloc[:, 1].astype(str)

# Merge the files based on the join key
merged = file1.merge(file2, on='key')
merged = merged.merge(file3, on='key')
merged = merged.merge(file4, on='key')

# Save the merged result to a new file
merged.to_csv('merged_4file_proba.tsv', sep='\t', index=False)
