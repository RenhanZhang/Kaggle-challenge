import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize as lb

def binarize(data, col_name):

    # construct the dict if it's not given
    unique_vals = sorted(set(data[col_name].values))
    cls = np.arange(1, 1 + len(unique_vals))
   
    # build the mapping dictionary
    mapping = {}
    for id, val in enumerate(unique_vals):
         mapping[val] = id+1

    label = data[col_name].map(mapping)
    chunk = pd.DataFrame(lb(label, classes=cls))
    chunk.columns = ['%s_%s' %(col_name, x) for x in chunk.columns]
    data = pd.concat([data, chunk], axis=1)

    # drop the orginal column
    data = data.drop(col_name, axis = 1)
    return data

def munge(fname):
    data = pd.read_csv(fname + '.csv')
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    
    if fname == 'train':
         m, std = {}, {}
    else:
         m, std = pd.read_csv('mean.csv'), pd.read_csv('std.csv')
         m.set_index('attr', inplace=True)
         std.set_index('attr', inplace=True)
         
    for col in cols:
         # only binarize columns with character values
         if col not in num_cols:
	      data = binarize(data, col)
         else:
              if col == 'Id' or col == 'Hazard':
                   continue

	      if fname == 'train':
                   u = data[col].mean(0)
                   sigma = data[col].std(0)
                   m[col] = u
                   std[col] = sigma
              else:
                   u = m.loc[col]['mean']
                   sigma = std.loc[col]['std']
              data[col] = (data[col] - u)/sigma

    if fname == 'train':    
         f = open('mean.csv', 'w')
         writer = csv.writer(f)
         writer.writerow(['attr', 'mean'])
         for k in sorted(m.keys()):
              writer.writerow([k, m[k]])
         f.close()

         f = open('std.csv', 'w')
         writer = csv.writer(f)
         writer.writerow(['attr','std'])
         for k in sorted(std.keys()):
              writer.writerow([k, std[k]]) 
         f.close()

    data.to_csv(fname+'_clean.csv', index=False)

    return data
if __name__ == '__main__':
    munge('train')
