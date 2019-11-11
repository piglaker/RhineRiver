import pandas as pd
import utils
import numpy as np


dir_data = 'river_data.csv'
dir_stations = 'river_stations.csv'

def make_dataset(dir_data, dir_stations):
    data, stations = pd.read_csv(dir_data), pd.read_csv(dir_stations)

    #s = [str(i) for i in set(stations['GRDC-No.'])]
    #s2 = set(stations['Nextdownstreamstation'])

    d = dict()

    for i in range(len(stations)):
        d[str(stations.iloc[i]['GRDC-No.'])] = stations.iloc[i]['Nextdownstreamstation']

    index = {'6335050':d['6335050']}

    print(d['6335060'])
    print(len(d))

    for i in d:
        if i in [j for j in index.values()] or d[i] in [p for p in index.keys()]:
            index[i] = d[i]


    print(index)
    print(len(index))



    dict_data = dict()

    for i in index.keys():
        dict_data[i] = []

    for i in range(len(data)):
        print(i)
        name = str(data.iloc[i]['station_no'])

        if name in index.keys():

            dict_data[name].append(np.array([data.iloc[i]['date']] +
                                   [data.iloc[i]['discharge']] +
                                   [data.iloc[i]['water_level']]))

    print(dict_data)
    
    for i in dict_data.keys():
        p = pd.DataFrame(dict_data[i])
        p.to_csv('./rhine/' + i +'.csv' )


if __name__ == '__main__':
    make_dataset(dir_data, dir_stations)

