import numpy as np
import pandas as pd
import os
import utils


def app():

    dir_data = 'river_data.csv'
    dir_stations = 'river_stations.csv'


    data, stations = pd.read_csv(dir_data), pd.read_csv(dir_stations)

    # s = [str(i) for i in set(stations['GRDC-No.'])]
    s = set(data['date'])

    d = dict()

    for i in range(len(stations)):
        d[str(stations.iloc[i]['GRDC-No.'])] = stations.iloc[i]['Nextdownstreamstation']

    index = {'6335050': d['6335050']}

    for i in d:
        if i in [j for j in index.values()] or d[i] in [p for p in index.keys()]:
            index[i] = d[i]

    dict_data = dict()
    for i in s:
        dict_data[i] = []

    for i in range(len(data)):
        print(i)
        if str(data.iloc[i]['station_no']) in index.keys():
            if data.iloc[i]['discharge'] != -999:

                if data.iloc[i]['water_level'] != -999:
                    dict_data[str(data.iloc[i]['date'])].append([data.iloc[i]['station_no'],
                                                    data.iloc[i]['discharge'],
                                                    data.iloc[i]['water_level']])

    print(dict_data)

    print(len(dict_data))
    print(max(dict_data.values()))

    for i in dict_data.keys():
        if len(dict_data[i])>0:
            p = pd.DataFrame(dict_data[i])
            p.to_csv('./stations/' + str(i) + '.csv')


if __name__ == '__main__':
    app()