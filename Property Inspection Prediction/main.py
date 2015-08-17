import numpy as np
import pandas as pd
from sklearn.svm import SVR
import sklearn
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import Ridge as rg
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import os.path
import re
import matplotlib.pyplot as plt

from data_munge import munge

if os.path.isfile('train_clean.csv'):
     train_data = pd.read_csv('train_clean.csv')
else:
     train_data = munge('train')

col_to_remove = ['T1_V10', 'T1_V11', 'T1_V13', 'T1_V14', 'T1_V17', 'T1_V5', 'T1_V6', 'T1_V9', 'T2_V10', 'T2_V11', 'T2_V12', 'T2_V13', 'T2_V3', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8']
y = train_data['Hazard']
X = train_data.drop(['Id', 'Hazard'],axis=1)
'''
for c in X.columns:
    for col in col_to_remove:
        if re.match(col, c):
            X = X.drop([c],axis=1)
            break
'''
X = pd.concat([pd.DataFrame(np.ones(len(y))), X], axis = 1)

base_estimator = GradientBoostingRegressor(n_estimators=1000, loss='ls', verbose = 2)
clf = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, oob_score=True, n_jobs=-1)
#clf = GradientBoostingRegressor(n_estimators=2000, loss='huber', verbose = 2)
#clf = RandomForestRegressor(n_estimators=5000, n_jobs=-1)
#clf = rg(alpha=0.1)
#clf = sklearn.linear_model.Lasso(alpha=0.1)
#clf = SVR()
#clf = RFECV(estimator=rg(alpha=0.1), step=1, cv=5)

clf.fit(X, y)
'''
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (default scoring of the estimator)")
plt.plot(range(1, len(clf.grid_scores_) + 1), clf.grid_scores_)
plt.savefig('cv.png')
'''
if os.path.isfile('test_clean.csv'):
     test_data = pd.read_csv('test_clean.csv')
else:
     test_data = munge('test')

Id = test_data['Id']
test_data = test_data[[col for col in test_data.columns if col != 'Id']]
'''
for c in test_data.columns:
    for col in col_to_remove:
        if re.match(col, c):
            test_data = test_data.drop([c],axis=1)
            break
'''
test_data = pd.concat([pd.DataFrame(np.ones(len(test_data))), test_data], axis=1)
predictions = clf.predict(test_data)
#predictions = np.round(predictions)
predictions = pd.concat([pd.DataFrame(Id), pd.DataFrame(predictions)], axis=1)
predictions.columns = ['Id', 'Hazard']

predictions.to_csv('submission.csv', index=False)



