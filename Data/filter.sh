#!/bin/bash

# Define input file path containing variant data
input_file="reva_human_38_uniq.txt"

# Define filtering criteria and output files
# Keys: Cell line names
# Values: Positive and negative p-value thresholds separated by space
# Format: ["CellLine"]="positive_threshold negative_threshold"
declare -A cell_lines=(
    ["K562"]="0.01 0.999"       # Positive: p < 0.01, Negative: p > 0.999
    ["HepG2"]="0.01 0.999"     # Positive: p < 0.01, Negative: p > 0.999
    ["GM18507"]="0.01 0.99"    # Positive: p < 0.01, Negative: p > 0.99
    ["GM12878"]="0.01 0.99"    # Positive: p < 0.01, Negative: p > 0.99
    ["HEK293T"]="0.01 0.99"    # Positive: p < 0.01, Negative: p > 0.99
    ["HEK293FT"]="0.01 0.99"   # Positive: p < 0.01, Negative: p > 0.99
    ["HaCaT"]="0.01 0.99"      # Positive: p < 0.01, Negative: p > 0.99
)

# Define cell lines requiring top N negative results and their counts
# Keys: Cell line names
# Values: Number of top negative results to keep
declare -A top_n_negative=(
    ["GM12878"]="1956"   # Keep top 1956 negative results for GM12878
    ["HEK293FT"]="641"   # Keep top 641 negative results for HEK293FT
)

# Process each cell line
for cell_line in "${!cell_lines[@]}"; do
    # Extract thresholds for current cell line
    IFS=' ' read -r pos_threshold neg_threshold <<< "${cell_lines[$cell_line]}"

    # Process positive results (unchanged criteria)
    # Filter logic:
    # 1. Find lines for current cell line
    # 2. Select rows with p-value < pos_threshold and not missing (.)
    # 3. Select positive class (column 16 == 1)
    # 4. Extract first 5 columns
    less "$input_file" | grep "$cell_line" | awk -v thresh="$pos_threshold" '$12 < thresh && $12 != "."' | awk '$16 == 1' | cut -f 1-5 > "${cell_line}_1.vcf"

    # Process negative results (modified criteria)
    # Filter logic:
    # 1. Find lines for current cell line
    # 2. Select rows with p-value > neg_threshold and not missing (.)
    # 3. Select negative class (column 16 == 0)
    # 4. Sort by p-value in descending order (largest p-values first)
    # 5. Extract first 5 columns
    negative_results=$(less "$input_file" | grep "$cell_line" | awk -v thresh="$neg_threshold" '$12 > thresh && $12 != "."' | awk '$16 == 0' | sort -k12,12gr | cut -f 1-5)

    # Handle top N selection for specified cell lines
    if [[ -n "${top_n_negative[$cell_line]}" ]]; then
        top_n="${top_n_negative[$cell_line]}"
        # Take the top N results after sorting
        echo "$negative_results" | head -n "$top_n" > "${cell_line}_0.vcf"
    else
        # Keep all qualified negative results
        echo "$negative_results" > "${cell_line}_0.vcf"
    fi
done
