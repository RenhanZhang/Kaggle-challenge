import pandas as pd
from sklearn.svm import SVR
from sklearn import cross_validation, grid_search
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge as rg
from sklearn.ensemble import GradientBoostingRegressor
import sklearn
from sklearn.externals import joblib
import re
import os

def display_unique():
    # check if the categorical columns in train and test data have the same unique values
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    num_cols = train._get_numeric_data().columns
    for col in train.columns:
        if col in num_cols:
            continue
        print '-------------%s----------------' % col
        ta = sorted(train[col].unique())
        te = sorted(test[col].unique())
        print ta == te
        print ta
        print te


def var_select():
    if os.path.exists('feature_selection_log.txt'):
        os.remove('feature_selection_log.txt')
    #clf = SVR()
    #clf = rg(alpha=0.1)
    clf = sklearn.linear_model.Lasso(alpha=0.1)
    min_features = 1

    #raw_cols = [u'T1_V1', u'T1_V2', u'T1_V3', u'T1_V4', u'T1_V5', u'T1_V6', u'T1_V7']
    raw_cols = [u'T1_V1', u'T1_V2', u'T1_V3', u'T1_V4', u'T1_V5', u'T1_V6', u'T1_V7', u'T1_V8', u'T1_V9', u'T1_V10', u'T1_V11', u'T1_V12', u'T1_V13', u'T1_V14', u'T1_V15', u'T1_V16', u'T1_V17', u'T2_V1', u'T2_V2', u'T2_V3', u'T2_V4', u'T2_V5', u'T2_V6', u'T2_V7', u'T2_V8', u'T2_V9', u'T2_V10', u'T2_V11', u'T2_V12', u'T2_V13', u'T2_V14', u'T2_V15']

    data = pd.read_csv('train_clean.csv')
    y = data['Hazard']
    X = data.drop(['Id', 'Hazard'], axis=1)

    dropped_cols = []
    remain_cols = set(raw_cols)
    assert len(remain_cols) == len(raw_cols)
    num_features = []
    its_scores = []
    while True:

        if len(remain_cols) <= min_features:
            break
        best_score = -1

        print 'Round %s' % (len(dropped_cols)+1)
        # find the worst col to drop
        for col in remain_cols:
            x = X[[c for c in X.columns if not re.match(col, c)]]
            score = cross_validation.cross_val_score(clf, x, y, cv=5).mean()

            if score > best_score:
                best_score = score
                worst_col = col
        
        dropped_cols.append('\''+worst_col+'\'')
        
        remain_cols.remove(worst_col)
        X = X[[c for c in X.columns if not re.match(worst_col, c)]]

        with open('feature_selection_log.txt', 'a') as f:
            f.write('\n--------Round %s, score %s -----------\n' % (len(dropped_cols), best_score))
            f.write(', '.join(sorted(dropped_cols)))
        
        num_features.append(len(remain_cols))
        its_scores.append(best_score)
    plt.plot(num_features, its_scores)
    plt.xlabel('number of features')
    plt.ylabel('accuracy score')
    plt.savefig('feature_selection.png')


def param_tuning(base_estimator, param, cv=5):
    data = pd.read_csv('train_clean.csv')
    X = data.drop(['Id', 'Hazard'], axis=1)
    y = data.Hazard
    clf = grid_search.GridSearchCV(estimator=base_estimator, param_grid=param, cv=cv, n_jobs=-1)
    clf.fit(X, y)
    return clf
'''
base_estimator = GradientBoostingRegressor(verbose=2)
param = {'loss': ['ls', 'lad', 'huber'], 'n_estimators': [100, 1000, 1500, 2000]}
clf = param_tuning(base_estimator, param)
joblib.dump(clf, 'cv_models/gradient_boosting.pkl') 
'''
