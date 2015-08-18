import numpy as np
import pandas as pd
from sklearn.svm import SVR
import sklearn
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
import os, re
from data_munge import munge

alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def stacking(estimators):
    # training
    predictions = []
    for estim in estimators:
        estim.fit(X, y)
        predictions.append(estim.predict(X))

    agg = RidgeCV(alphas=alphas, cv=5, normalize=True, fit_intercept=True)         # aggregator
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


def drop_cols(data, cols_to_drop):
    for col in data.columns:
        for dc in cols_to_drop:
            if re.match(dc, col):
                data.drop(col, axis=1, inplace=True)
                break


def singly_predict(clf, trainX, trainY, testX):
    clf.fit(trainX, trainY)
    results = clf.predict(testX)
    write_results(results)


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

Id = test_data['Id']
test_data.drop('Id', axis=1, inplace=True)

# drop cols
Dropcols = ['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10']
drop_cols(X, Dropcols)
drop_cols(test_data, Dropcols)

gbr = GradientBoostingRegressor(n_estimators=1000, loss='lad', verbose=2)
rfr = RandomForestRegressor(n_estimators=2000, n_jobs=3, verbose=2)
etr = ExtraTreesRegressor(n_estimators=2000, n_jobs=3, verbose=2)

bagged_gbr = BaggingRegressor(base_estimator=gbr, n_estimators=5, oob_score=True, n_jobs=3)
bagged_rfr = BaggingRegressor(base_estimator=rfr, n_estimators=5, oob_score=True, n_jobs=3)
bagged_etr = BaggingRegressor(base_estimator=etr, n_estimators=5, oob_score=True, n_jobs=3)

# ridge regressor
rg = RidgeCV(alphas=alphas, cv=5, normalize=True, fit_intercept=True)
rg.fit(X, y)
#clf = GradientBoostingRegressor(n_estimators=100, loss='ls', verbose=2)
#rfr = RandomForestRegressor(n_estimators=5000, n_jobs=-1)
#clf = rg(alpha=0.1)
#clf = sklearn.linear_model.Lasso(alpha=0.1)
#clf = SVR()
#clf = RFECV(estimator=rg(alpha=0.1), step=1, cv=5)
#clf.fit(X, y)

stacking([rg, bagged_gbr, bagged_rfr, bagged_etr])

