import pandas as pd

df = pd.read_csv('output_ind_test.csv')

data = df.iloc[:, 12:]

df['pred'] = (data.abs() > 0.3).any(axis=1)

df['pred'] = df['pred'].astype(int)

df.to_csv('expecto_ind_test_pred.csv', index=False)
