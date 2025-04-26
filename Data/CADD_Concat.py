import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Read the two CSV files
df0 = pd.read_csv('CADD_no_anno_0.csv', sep=',')
df1 = pd.read_csv('CADD_no_anno_1.csv', sep=',')

# Combine the data
combined_df = pd.concat([df0, df1], ignore_index=True)

# Shuffle the combined data (using the same random seed for consistency)
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled result to a new TSV file
shuffled_df.to_csv('CADD_no_anno.tsv', sep='\t', index=False)
