#!/bin/bash

# Define the path of the input file
input_file="reva_human_38_uniq.txt"

# Define the filtering conditions and output files
# The keys are cell lines, and the values are positive and negative p-value thresholds separated by a space
declare -A cell_lines=(
    ["K562"]="0.01 0.001"
    ["HepG2"]="0.01 0.001"
    ["GM18507"]="0.01 0.01"
    ["GM12878"]="0.01 0.01"
    ["HEK293T"]="0.01 0.01"
    ["HEK293FT"]="0.01 0.01"
    ["HaCaT"]="0.01 0.01"
)

# Define the cell lines for which the top N negative results are to be taken and the corresponding number
declare -A top_n_negative=(
    ["GM12878"]="1956"
    ["HEK293FT"]="641"
)

# Loop through each cell line
for cell_line in "${!cell_lines[@]}"; do
    # Get the positive and negative p-value thresholds
    IFS=' ' read -r pos_threshold neg_threshold <<< "${cell_lines[$cell_line]}"

    # Filter positive results
    # Read the input file, filter lines containing the current cell line, 
    # select lines where the p-value is less than the positive threshold and not equal to ".",
    # then select lines where the 16th column is equal to 1 (positive results),
    # and finally extract the first 5 columns and save them to a file ending with "_1.vcf"
    less "$input_file" | grep "$cell_line" | awk -v thresh="$pos_threshold" '{if ($12 < thresh && $12 != ".") print $0}' | awk '{if ($16 == 1) print $0}' | cut -f 1-5 > "${cell_line}_1.vcf"

    # Filter negative results
    negative_results=$(less "$input_file" | grep "$cell_line" | awk -v thresh="$neg_threshold" '{if ($12 < thresh && $12 != ".") print $0}' | awk '{if ($16 == 0) print $0}' | cut -f 1-5)

    # Check if the top N negative results need to be taken
    if [[ -n "${top_n_negative[$cell_line]}" ]]; then
        top_n="${top_n_negative[$cell_line]}"
        echo "$negative_results" | head -n "$top_n" > "${cell_line}_0.vcf"
    else
        echo "$negative_results" > "${cell_line}_0.vcf"
    fi
done
