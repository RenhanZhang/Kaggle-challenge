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

def stacking(estimators):
    # training
    predictions = []
    for estim in estimators:
        estim.fit(X, y)
        predictions.append(estim.predict(X))
    agg = rg(alpha=0.1)         # aggregator
    agg.fit(np.array(predictions).T, y)

    # test
    predictions = []
    for estim in estimators:
        predictions.append(estim.predict(test_data))

    predictions = agg.predict(np.array(predictions).T)
    write_results(predictions)

def write_results(result):
    result = pd.concat([pd.DataFrame(Id), pd.DataFrame(result)], axis=1)
    result.columns = ['Id', 'Hazard']
    result.to_csv('submission.csv', index=False)


if os.path.isfile('train_clean.csv'):
    train_data = pd.read_csv('train_clean.csv')
else:
    train_data = munge('train')

if os.path.isfile('test_clean.csv'):
    test_data = pd.read_csv('test_clean.csv')
else:
    test_data = munge('test')

y = train_data['Hazard']
X = train_data.drop(['Id', 'Hazard'], axis=1)
X = pd.concat([pd.DataFrame(np.ones(len(y))), X], axis=1)

Id = test_data['Id']
test_data = test_data[[col for col in test_data.columns if col != 'Id']]
test_data = pd.concat([pd.DataFrame(np.ones(len(test_data))), test_data], axis=1)

gbr = GradientBoostingRegressor(n_estimators=1000, loss='ls', verbose=2)
#clf = BaggingRegressor(base_estimator=base_estimator, n_estimators=5, oob_score=True, n_jobs=-1)
#clf = GradientBoostingRegressor(n_estimators=100, loss='ls', verbose=2)
rfr = RandomForestRegressor(n_estimators=5000, n_jobs=-1)
#clf = rg(alpha=0.1)
#clf = sklearn.linear_model.Lasso(alpha=0.1)
#clf = SVR()
#clf = RFECV(estimator=rg(alpha=0.1), step=1, cv=5)
#clf.fit(X, y)

stacking([BaggingRegressor(base_estimator=gbr, n_estimators=5, oob_score=True, n_jobs=-1),
          BaggingRegressor(base_estimator=rfr, n_estimators=5, oob_score=True, n_jobs=-1),
          SVR(), rg(alpha=0.1)])

