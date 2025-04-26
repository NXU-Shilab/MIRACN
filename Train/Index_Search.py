import pandas as pd

from train_nois import train_5f

fold_results, cell_type_accuracies, functionality_aucs = train_5f()

fold_results_df = pd.concat([pd.DataFrame([fold_results]) for fold_results in fold_results])
fold_results_df.to_csv('cnn_MIRACN_{}_fold_results_search_new.csv'.format(1993), header=True, index=False)
