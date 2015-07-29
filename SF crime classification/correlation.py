import pandas as pd
import matplotlib.pyplot as plt
import os
import re
data = pd.read_csv('processed.csv', header=0)
data.head()

#categories = list(set(data.Category.values))
#gp = data.groupby('Category').size()

def correlation(data, feature1, feature2):
    '''
       explore the relationship between feature1 and feature2
       by plot hisogram of feature2 for each of unique values in feature1
    '''
    dir = feature1 + '_' + feature2 + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    unique_vals = set(data[feature1].values)

    for val in unique_vals:
        slice = data[data[feature1]==val]
        slice = slice[feature2].value_counts().sort_index()
        slice.plot(kind='bar')

        plt.xlabel(feature2)
        plt.ylabel(feature1)
        val = re.sub('\W', ' ', val)
        plt.title(val)
        plt.savefig(dir + val + '_' + feature2 + '.png')
        plt.close()

correlation(data, 'Category', 'DayOfWeekNo')
correlation(data, 'Category', 'Month')
correlation(data, 'Category', 'PdDistrict')
'''
for i in range(len(categories)):
    cate = categories[i]

    plt.scatter(data[data.Category==cate].X, data[data.Category==cate].Y)
    size = str(gp.ix[cate])
    cat = re.sub('\W+', '', cate)
    plt.title(cate + '_' + size)
    plt.savefig('images/'+cat + '.png')

'''
'''
for i in range(len(categories)):
    cate = categories[i]
    slice = data[data.Category==cate]

    plt.hist(data[data.Category==cate]['DayOfWeek'].values)
    cat = re.sub('\W+', '', cate)
    plt.title(cate)
    plt.savefig('time correlation/weekwise/'+cat + '.png')
    plt.close()

plt.hist(data.DayOfWeek)
'''

