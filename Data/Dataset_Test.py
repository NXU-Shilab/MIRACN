import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def dataset_make_train():
    file1 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_1.csv')

    file2 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_0.csv')

    file1['label'] = 1
    file2['label'] = 0

    data = pd.concat([file1, file2], ignore_index=True)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    seqclass = data["seqclass_max_absdiff"]
    label = data["label"]
    selected_columns = ['GM12878_avg', 'GM18507_avg', 'HaCaT_avg', 'HEK293FT_avg', 'HEK293T_avg', 'HepG2_avg',
                        'K562_avg']
    sei_avg = data[selected_columns]


    X = data.drop(
        ["seqclass_max_absdiff", "index", "chrom", "pos", "name", "ref", "alt", "strand", "ref_match", "contains_unk",
         "cell_line", "label", 'GM12878_avg', 'GM18507_avg', 'HaCaT_avg', 'HEK293FT_avg', 'HEK293T_avg', 'HepG2_avg',
         'K562_avg'], axis=1)
    y_cell_type = data['cell_line']
    y_functionality = data['label']

    x_train_temp, x_test, y_train_cell_type_temp, y_test_cell_type, y_train_functionality_temp, y_test_functionality = train_test_split(
        X, y_cell_type, y_functionality, test_size=0.1, random_state=42)

    num_classes = 7
    y_cell_type = to_categorical(np.array(y_cell_type - 1), num_classes)
    # y_train_cell_type = to_categorical(np.array(y_train_cell_type - 1), num_classes)
    # y_test_cell_type = to_categorical(np.array(y_test_cell_type - 1), num_classes)
    y_test_cell_type = to_categorical(np.array(y_test_cell_type - 1), num_classes)



    return x_train_temp,y_train_cell_type_temp,y_train_functionality_temp

def dataset_make_test():
    file1 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_1.csv')

    file2 = pd.read_csv('/mnt/data0/users/lizy/Sei2/sei-framework-main/Positive/multi_1_lable1/multi_avg_0.csv')

    file1['label'] = 1
    file2['label'] = 0

    data = pd.concat([file1, file2], ignore_index=True)
    data = data.drop_duplicates(subset=data.columns[2:7])

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    seqclass = data["seqclass_max_absdiff"]
    label = data["label"]
    selected_columns = ['GM12878_avg', 'GM18507_avg', 'HaCaT_avg', 'HEK293FT_avg', 'HEK293T_avg', 'HepG2_avg',
                        'K562_avg']
    sei_avg = data[selected_columns]


    X = data.drop(
        ["seqclass_max_absdiff", "index", "chrom", "pos", "name", "ref", "alt", "strand", "ref_match", "contains_unk",
         "cell_line", "label", 'GM12878_avg', 'GM18507_avg', 'HaCaT_avg', 'HEK293FT_avg', 'HEK293T_avg', 'HepG2_avg',
         'K562_avg'], axis=1)
    y_cell_type = data['cell_line']
    y_functionality = data['label']

    x_train_temp, x_test, y_train_cell_type_temp, y_test_cell_type, y_train_functionality_temp, y_test_functionality, sei_avg_train, sei_avg_test = train_test_split(
        X, y_cell_type, label, sei_avg, test_size=0.1, random_state=42)

    num_classes = 7
    y_cell_type = to_categorical(np.array(y_cell_type - 1), num_classes)
    # y_train_cell_type = to_categorical(np.array(y_train_cell_type - 1), num_classes)
    # y_test_cell_type = to_categorical(np.array(y_test_cell_type - 1), num_classes)
    y_test_cell_type = to_categorical(np.array(y_test_cell_type - 1), num_classes)

    return x_test, y_test_cell_type, y_test_functionality,seqclass,label,sei_avg_test,y_cell_type
def dataset_make_CADD():


    cadd_0 = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/test/test_CADD_0.csv')
    cadd_1 = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/test/test_CADD_1.csv')

    cadd_0['label'] = 0
    cadd_1['label'] = 1

    data_cadd = pd.concat([cadd_0, cadd_1], ignore_index=True)

    data_cadd = data_cadd.sample(frac=1, random_state=42).reset_index(drop=True)

    cadd = data_cadd["PHRED"]
    label_cadd = data_cadd["label"]
    return cadd, label_cadd
def dataset_make_Expecto():
    file = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/my_expecto/Expecto/expecto_pred.csv')
    label_expecto = file['5']
    pre_expecto = file['pred']
    return label_expecto, pre_expecto
def dataset_make_DVAR():
    file = pd.read_csv('/mnt/data0/users/lizy/pycharm_project/test/DVAR/DVAR_test.tsv', sep='\t')
    label_DVAR =  file.iloc[:, 3]
    probability_DVAR = file.iloc[:, 2]
    return probability_DVAR, label_DVAR
