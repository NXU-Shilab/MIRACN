# MIRACN

Welcome to the ``` MIRACN ``` framework repository!
``` MIRACN ``` is a multi-task learning framework based on a residual convolutional neural network, trained using the MPRA dataset. It performs two tasks simultaneously: predicting functional regulatory variants and predicting the cell lines in which these variants are active. Additionally, it explores the functional characteristics of cell line-specific variants.

# Requirements

Please create a new conda environment specifically for running ``` MIRACN ``` (e.g. ``` conda create --MIRACN python=3.9.18 ```), install the packages listed in the ``` requirements.txt ``` file. Install with conda or pip (e.g. ``` conda install pandas==2.2.3 ```).

# Data

we utilized Massively Parallel Reporter Assays (MPRA), which contains over 11.8 million experimentally validated expression regulatory variants across multiple human genome cell lines from REVA database (https://reva.gao-lab.org/download.php). Specifically, we downloaded and used the REVA version 1.1.1 GRCh38 unique dataset for our analysis. 

# Feature
We used the ```sei-framework``` model for feature annotation of variants (https://github.com/FunctionLab/sei-framework?tab=readme-ov-file). To use cell line-specific functional prediction, please install the sei-framework model and place the ```sei-framework-main``` directory in the same directory level as the ```Prediction``` directory to enable calling its feature annotation scripts.

# Predicting

The following scripts can be used to obtain MIRACN predictions for variants.

Example usage:

```
sh predict.sh <vcf_file> <output_dir> [--cuda] <output_file>
```

Arguments:

* ```<vcf_file>```: VCF file

* ```<output_dir>```: Path to feature annotation output directory.

* ```--cuda```: Optional, use this flag if running on a CUDA-enabled GPU.

* ```<output_file>```: Path to the prediction probability output file.

The output will be saved to output_file. The first five columns of the output file will remain the same as the original VCF files. Additional columns will include functional predictions and cell-type-specific predictions for enhanced variant interpretation.

# Training

You can train the model through the following process.

### Feature annotation

Feature annotation, please refer to Chen KM, Wong AK, Troyanskaya OG, et al. "A sequence-based global map of regulatory activity for deciphering human genetics" (https://github.com/FunctionLab/sei-framework).

To obtain 21,907 chromatin profile features and 40 sequence class features for each variant, follow and execute the following two commands mentioned in the Sei repository's README.md:

``` sh 1_variant_effect_prediction.sh <vcf> <hg> <output-dir> [--cuda] ```

``` sh 2_varianteffect_sc_score.sh <ref-fp> <alt-fp> <output-dir> [--no-tsv] ```

### Feature selection

Annotating variants features, extracting corresponding features.
 
Example usage:

```
sh feature_extraction.sh sorted.example.chromatin_profile_diffs.tsv example.tsv
```

Arguments:

sh your_script_name.py input_filename.tsv output_filename.tsv

### Training multi-task model

After annotating the features, train a multi-task model.We provide model training and parameter search code. Please place these files in the same directory as ```Dataset_Train.py```, and make sure to update the training dataset path in ```Dataset_Train.py``` with your own path.

Example usage:

```
python Train.py
```
```
python Index_Search.py
```

Arguments:

python your_script_name.py

### Predicting Rare Variant & GTEx data

Predict two independent testing datasets. Please ensure that the code and ```Dataset_Test.py``` file are placed in the same directory to properly call functions from ```Dataset_Test.py```.The output files will be saved in current directory.

Example usage:

```
python Predict_Rare_Variants.py
```
```
python Predict_GTEx.py
```

Arguments:

python your_script_name.py

# Help

Please post in the Github issues or e-mail Fangyuan Shi (shify@nxu.edu.cn) with any questions about the repository, requests for more data, additional information about the results, etc.



