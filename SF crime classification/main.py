import xgboost as xgb
import pandas as pd
from data_munge import munge
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib

#features = ['DayOfWeekNo', 'PdDistrict_ID', 'Year', 'Month', 'Day', 'Hour', 'X_quan', 'Y_quan', 'Street1_ID', 'Street2_ID']

print 'preparing training data...'
if not os.path.exists('train_clean.csv'):
    train_raw = pd.read_csv('train.csv')
    train_prepared = munge(train_raw, 'train_clean.csv')
else:
    train_prepared = pd.read_csv('train_clean.csv')

neg_fea = [u'X_quan', u'Y_quan', u'X', u'Y', u'Category', u'Descript', u'DayOfWeek', u'PdDistrict', u'Resolution', u'Category_ID',
           u'Address', u'PdDistrict_ID', u'DayOfWeekNo', u'Day', u'Month', u'Year', u'Hour', u'Dates']

y = train_prepared['Category_ID']
X = train_prepared.drop(neg_fea, 1)
X = X.drop(X.columns[0], 1)

# free up memory occupied by train_prepared
del train_prepared
train_prepared = None

print 'Fitting...'
start_time = time.time()

agg_predictions = []

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X,y)
agg_predictions.append(pd.DataFrame(rfc.predict_proba(X)))

gbc = GradientBoostingClassifier(n_estimators=1000, loss='lad', verbose=2)
gbc.fit(X,y)
agg_predictions.append(pd.DataFrame(gbc.predict_proba(X)))

agg_predictions = pd.concat(agg_predictions, axis=1)

aggregator = LogisticRegression(penalty='l2', C=1, multi_class='multinomial')
aggregator.fit(agg_predictions, )
#clf = LogisticRegression(penalty='l2', C=1)
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=0).fit(X,y)
clf.fit(X,y)

#joblib.dump(clf, 'model.pkl')

# free up X and y
del X
del y
X = None
y = None

print 'preparing testing data...'

if not os.path.exists('test_clean.csv'):
    test_raw = pd.read_csv('test.csv')
    test_prepared = munge(test_raw, 'test_clean.csv')
else:
    test_prepared = pd.read_csv('test_clean.csv')

neg_fea = [u'X_quan', u'Y_quan', u'X', u'Y', u'Id', u'DayOfWeek', u'PdDistrict', u'Address', u'PdDistrict_ID', u'DayOfWeekNo',
           u'Day', u'Month', u'Year', u'Hour', u'Dates']
test_prepared = test_prepared.drop(neg_fea, 1)
test_prepared = test_prepared.drop(test_prepared.columns[0], 1)
predictions = clf.predict_proba(test_prepared)
predictions = pd.DataFrame(predictions.astype(float))
predictions = pd.concat([pd.DataFrame(np.arange(len(predictions))), predictions], 1)
ctgr = pd.read_csv('Category_ID.csv')

ctgr_txt = ctgr.Category.values.tolist()
ctgr_txt.insert(0, 'ID')
predictions.columns = ctgr_txt
predictions.set_index('ID', inplace=True)
predictions = predictions.sort_index(axis=1)
predictions.to_csv('submission.csv', float_format='%.6f')
'''
predictions = clf.predict(test_prepared)

del clf
clf = None
del test_prepared
test_prepared = None

N = len(predictions)
raw_predictions = np.hstack(((np.arange(N)).reshape([N,1]), predictions.reshape([N,1])))
raw_predictions = pd.DataFrame(raw_predictions)
raw_predictions.to_csv('raw_predictions')
N = len(predictions)
ctgr = pd.read_csv('Category_ID.csv')

result = np.zeros([N, 39])

for i in xrange(N):
    result[i, predictions[i]-1] = 1

result = np.hstack(((np.arange(N)).reshape([N,1]), result))
ctgr_txt = (ctgr.Category.values).tolist()
ctgr_txt.insert(0, 'ID')

result = result.astype(int)
result = pd.DataFrame(result)
result.columns = ctgr_txt
result.set_index('ID', inplace=True)
result = result.sort_index(axis=1)
result.to_csv('submission.csv')
'''
