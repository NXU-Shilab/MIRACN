import pandas as pd

# Load file1 and file2
file1 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_1.csv')
file2 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_0.csv')

# Load the first two columns of the GTEx file
gtex = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/independent_test_GTEx/Sei_new/GTEx_MIRACN_fea_cp.tsv', delim_whitespace=True, usecols=[2, 3], names=['chr', 'pos'])

# Combine the third and fourth columns of file1 and file2 as the key
file1['key'] = file1.iloc[:, 2].astype(str) + '_' + file1.iloc[:, 3].astype(str)
file2['key'] = file2.iloc[:, 2].astype(str) + '_' + file2.iloc[:, 3].astype(str)

# Combine the first two columns of gtex as the key
gtex['key'] = gtex['chr'].astype(str) + '_' + gtex['pos'].astype(str)

# Find the rows with the same key in file1 and gtex
file1_matches = file1[file1['key'].isin(gtex['key'])]
file2_matches = file2[file2['key'].isin(gtex['key'])]

# Get the set of matching keys
matching_keys = set(file1_matches['key']).union(set(file2_matches['key']))

# Get the indices of the matching rows from the gtex file
gtex_matches_index = gtex[gtex['key'].isin(matching_keys)].index

gtex_matches_index = gtex_matches_index + 1
# Save the indices of the matching rows to a file
matching_indices_df = pd.DataFrame({'index': gtex_matches_index})

# Save the indices of the matching rows to a file
matching_indices_df.to_csv('/mnt/data0/users/lizy/pycharm_project/independent_test_GTEx/Sei_new/matching_indices.csv', index=False)

# Check if there are matching results and save the output file or report information
if not matching_indices_df.empty:
    print("The indices of the matching rows have been saved to the file: matching_indices.csv")
else:
    print("No rows with the same key were found.")    
