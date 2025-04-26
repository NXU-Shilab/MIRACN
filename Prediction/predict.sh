#!/bin/bash

# Check if the number of input arguments is either 3 or 4
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <vcf_file> <output_dir> [--cuda] <feature_file>"
    exit 1
fi

# Get the command-line arguments
vcf_file=$1
output_dir=$2
cuda=""
feature_file=""

# Check if the third argument is --cuda
if [ "$3" == "--cuda" ]; then
    cuda="--cuda"
    feature_file=$4  # If CUDA is enabled, the fourth argument is the feature_file
else
    feature_file=$3  # If CUDA is not enabled, the third argument is the feature_file
fi

# Define the path to the sei-framework-main directory
sei_framework_dir="../sei-framework-main"

# Switch to the sei-framework-main directory and execute the first command
pushd "$sei_framework_dir" > /dev/null

sh 1_variant_effect_prediction.sh "$vcf_file" hg38 "../MIRACN/$output_dir" "$cuda"
if [ $? -ne 0 ]; then
    echo "The first command failed"
    popd > /dev/null
    exit 1
fi

# Remove the file extension of the vcf file
vcf_file_base="${vcf_file%.*}"

# Execute the second command
sh 2_varianteffect_sc_score.sh "../MIRACN/$output_dir/chromatin-profiles-hdf5/$vcf_file_base.ref_predictions.h5" "../MIRACN/$output_dir/chromatin-profiles-hdf5/$vcf_file_base.alt_predictions.h5" "../MIRACN/$output_dir"
if [ $? -ne 0 ]; then
    echo "The second command failed"
    popd > /dev/null
    exit 1
fi


# Return to the original directory and execute the feature_extraction.sh script
popd > /dev/null
sh "feature_extraction.sh" "$output_dir/sorted.$vcf_file_base.chromatin_profile_diffs.tsv" "$feature_file"
if [ $? -ne 0 ]; then
    echo "The third command failed"
    exit 1
fi

# Execute the predict.py script
python predict.py "$feature_file"
if [ $? -ne 0 ]; then
    echo "The fourth command failed"
    exit 1
fi

echo "All steps completed successfully."
