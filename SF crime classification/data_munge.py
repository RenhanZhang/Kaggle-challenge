import pandas as pd
from dateutil.parser import parse
import csv
import re
import editdistance
from sklearn.preprocessing import label_binarize as lb

x_min = -122.51364206429
x_max = -122.36493749408001
y_min = 37.707879022413501
y_max = 37.819975492297004

def quantize(val, min, max, resolution):
    label = 0
    step = (max-min)/resolution
    temp = min
    while (temp < val) and (temp < max):
        label = label + 1
        temp = temp + step

    return label

def digitialize(data, col_name, dict={}, map_func = None):
    '''
       assign unique numerical values to entries in data.colname
       write the value:id to csv file for later use
       return the digitalized data
    '''

    # construct the dict if it's not given
    if not dict:
        unique_vals = sorted(set(data[col_name].values))
        for id, val in enumerate(unique_vals):
            dict[val] = id+1

    # write the csv file if required

    with open(col_name+'_ID.csv', 'w') as f:
         writer = csv.writer(f)
         writer.writerow(['ID',col_name])
         for k, v in sorted(list(dict.iteritems()), key=lambda x:x[1]):
              writer.writerow([v, k])

    if map_func is None:
        data[col_name+'_ID'] = data[col_name].map(dict)
    else:
        data[col_name+'_ID'] = data[col_name].apply(lambda x: map_func(x, dict))

    return data

def extract_street(str):
    '''
       extract two streets from the address
       duplicate if there is only one streets
    '''
    pat = '(\w+ (EX|RW|BUFANO|WK|MAR|PARK|FERLINGHETTI|I\-80|AL|PL|HY|TR|CT|LN|CR|HWY|TER|PZ|DR|BL|ST|AV|RD|WY|WAY))'
    l = re.findall(pat, str)
    if len(l) == 1:
        return [l[0][0], l[0][0]]
    elif len(l) > 1:
        return [l[0][0], l[1][0]]
    else: print str

def find_strt_label(street, dict):
    '''
    find a numerical label for street based on dict,
    if there is no entry for street in the label,
    return the label of the street with min editdistance to the input street
    '''
    if street in dict.keys():
        return dict[street]

    min_dist = 100000

    for s in dict.keys():
        edit_dist = editdistance.eval(street, s)
        if edit_dist < min_dist:
            min_dist = edit_dist
            nearest_street = s

    return dict[nearest_street]


def munge(data, fname):

    if fname == 'train_clean.csv':
        data = digitialize(data, 'Category', dict={})

    # give street a label
    #data['Streets'] = data.Address.apply(extract_street)
    #data['Street1'] = data.Streets.apply(lambda x: x[0])
    #data['Street2'] = data.Streets.apply(lambda x: x[1])
    
    '''
    if fname == 'train_clean.csv':
        streets_label = {}
        unique_streets = set([str for sublist in data.Streets.tolist() for str in sublist])
        for i, t in zip(xrange(len(unique_streets)), unique_streets):
            streets_label[t] = i
    elif fname == 'test_clean.csv':
        # read dict from the file
        streets_label = pd.read_csv('Street1_ID.csv')
        streets_label = streets_label.set_index('Street1')
        streets_label = streets_label.to_dict()
        streets_label = streets_label['ID']

    data = digitialize(data, 'Street1', dict = streets_label, map_func=find_strt_label)
    data = digitialize(data, 'Street2', dict = streets_label, map_func=find_strt_label)
    '''
    data['X_quan'] = data['X'].apply(lambda x: quantize(x, x_min, x_max, 30))
    data['Y_quan'] = data['Y'].apply(lambda x: quantize(x, y_min, y_max, 20))
    data = digitialize(data, 'PdDistrict', dict={})

    # digitalized day of week
    day_dict = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    data['DayOfWeekNo'] = data.DayOfWeek.map(day_dict)
    time_objs = data.Dates.apply(parse)

    # digitalize time
    data['Day'] = time_objs.apply(lambda x: x.date().day)
    data['Month'] = time_objs.apply(lambda x: x.date().month)
    data['Year'] = time_objs.apply(lambda x: x.date().year)
    data['Hour'] = time_objs.apply(lambda x: x.hour)

    for field in sorted(['PdDistrict_ID', 'DayOfWeekNo', 'Day', 'Month', 'Year', 'Hour', 'X_quan', 'Y_quan']):
        unique = set(data[field])
	print field, len(unique)
        print unique
        cls = range(1, len(unique)+1)
        if field is 'Year':
            cls = range(2003, 2016)
        data = pd.concat([data, pd.DataFrame(lb(data[field], classes=cls))], axis=1)
    #day_binarized = label_binarize(data['Day'])
    data.to_csv(fname, index=False)
    return data

def main():
    data = pd.read_csv('train.csv')
    return munge(data,True)
if __name__ == '__main__':
    main()
