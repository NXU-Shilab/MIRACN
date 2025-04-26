import pandas as pd

# Read the data
variants_df = pd.read_csv('merged_4file_proba_new.tsv', sep='\t')

# Set the thresholds
threshold_MIRACN = 0.1580081582069397
threshold_CADD = 10
threshold_Sei = 0.0778954124895996
threshold_DVAR = 0.6336
threshold_Expecto = 0.3

# Determine the prediction results of each tool according to the thresholds
variants_df['Proba_pred'] = variants_df['Proba'] >= threshold_MIRACN
variants_df['Seqclass_pred'] = variants_df['Seqclass'] >= threshold_Sei
variants_df['PHRED_pred'] = variants_df['PHRED'] >= threshold_CADD
variants_df['DVAR-score_pred'] = variants_df['DVAR-score'] >= threshold_DVAR
variants_df['proba_Expecto_pred'] = variants_df['proba_Expecto'] >= threshold_Expecto

# Compare the true labels with the prediction results of the tools
variants_df['Proba_correct'] = variants_df['label'] == variants_df['Proba_pred']
variants_df['Seqclass_correct'] = variants_df['label'] == variants_df['Seqclass_pred']
variants_df['PHRED_correct'] = variants_df['label'] == variants_df['PHRED_pred']
variants_df['DVAR-score_correct'] = variants_df['label'] == variants_df['DVAR-score_pred']
variants_df['proba_Expecto_correct'] = variants_df['label'] == variants_df['proba_Expecto_pred']

# Find the variants that are correctly predicted only by MIRACN while all other tools predict them wrongly
miracn_correct_only = variants_df[
    (variants_df['Proba_correct']) &  # MIRACN is correct
    (~variants_df['Seqclass_correct']) &  # Sei is wrong
    (~variants_df['PHRED_correct']) &  # CADD is wrong
    (~variants_df['DVAR-score_correct']) &  # DVAR is wrong
    (~variants_df['proba_Expecto_correct'])  # Expecto is wrong
]

# Save the results to a file
miracn_correct_only.to_csv('../end_product/miracn_correct_only_proba.tsv', sep='\t', index=False)

print("The results have been saved to the file ../end_product/miracn_correct_only.tsv")
