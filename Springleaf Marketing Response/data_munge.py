import pandas as pd

def munge(fname):
    cols_to_drop = ['ID', 'VAR_0044']
    data = pd.read_csv(fname+'.csv')

    # drop irrelavent columns
    data.drop(cols_to_drop, inplace=True, axis=1)

    num_cols = data._get_numeric_data().columns

    cat_cols = set(data.columns).difference(num_cols)
    cat_cols.remove('target')   # prevent the label from being munged
    cat_cols = sorted(list(cat_cols))

    for col in cat_cols:

