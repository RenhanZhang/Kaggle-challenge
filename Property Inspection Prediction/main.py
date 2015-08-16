import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge as rg
from sklearn.feature_selection import RFECV
import os.path
import matplotlib.pyplot as plt
# feature selection
from data_munge import munge

if os.path.isfile('train_clean.csv'):
     train_data = pd.read_csv('train_clean.csv')
else:
     train_data = munge('train')


y = train_data['Hazard']
X = train_data.drop(['Id', 'Hazard'],axis=1)
X = pd.concat([pd.DataFrame(np.ones(len(y))), X], axis = 1)

#clf = rg(alpha=0)
#clf = SVR()
clf = RFECV(estimator=rg(alpha=0.1), step=1, cv=5)
clf.fit(X, y)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (default scoring of the estimator)")
plt.plot(range(1, len(clf.grid_scores_) + 1), clf.grid_scores_)
plt.savefig('cv.png')

if os.path.isfile('test_clean.csv'):
     test_data = pd.read_csv('test_clean.csv')
else:
     test_data = munge('test')

Id = test_data['Id']
test_data = test_data[[col for col in test_data.columns if col != 'Id']]
test_data = pd.concat([pd.DataFrame(np.ones(len(test_data))), test_data], axis=1)
predictions = clf.predict(test_data)
#predictions = np.round(predictions)
predictions = pd.concat([pd.DataFrame(Id), pd.DataFrame(predictions)], axis=1)
predictions.columns = ['Id', 'Hazard']

predictions.to_csv('submission.csv', index=False)



