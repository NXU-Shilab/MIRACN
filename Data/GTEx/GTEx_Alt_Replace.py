import pandas as pd

# Read the VCF file
vcf_file_path = '../independent_test_GTEx/output_modified/GTEx_use_label_deleted_37.vcf'
vcf_df = pd.read_csv(vcf_file_path, sep='\t', header=None)

# Read the TXT file
txt_file_path = '../independent_test_GTEx/GTEx-variant-data.txt'
txt_df = pd.read_csv(txt_file_path, sep='\t')

# Delete the 'chr' in the first column of the TXT file and merge it with the second column to form a new key
txt_df['chrom'] = txt_df['chrom'].str.replace('chr', '', regex=False)
txt_df['key'] = txt_df['chrom'].astype(str) + ':' + txt_df['chrom_pos'].astype(str)

# Create a dictionary for quick lookup
value_dict = dict(zip(txt_df['key'], txt_df['alt']))

# Merge the first and second columns of the VCF file into a new key
vcf_df['key'] = vcf_df[0].astype(str) + ':' + vcf_df[1].astype(str)

# Replace the values in the fifth column of the VCF file
vcf_df[4] = vcf_df['key'].map(value_dict)

# Remove the redundant key column
vcf_df.drop(columns=['key'], inplace=True)

# Save the updated VCF file
updated_vcf_file_path = '../independent_test_GTEx/updated_GTEx_use_label_deleted_38.vcf'
vcf_df.to_csv(updated_vcf_file_path, sep='\t', header=None, index=False)

print(f'Updated VCF file saved to {updated_vcf_file_path}')
