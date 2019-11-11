import xlrd
import math
import datetime
import pandas as pd
import numpy as np

def readFile():
    dir_answer = []
    dir_set = set()

    ans =[]
    v = []

    for dir_number in range(23743, 23743 + 17532):
        Date = xlrd.xldate_as_tuple(dir_number, 0)
        subPath = r'/%04d-%02d-%02d.csv' % (Date[0], Date[1], Date[2])

        try:
            pd.read_csv(r'./stations' + subPath)
            v.append(subPath)
        except:
            ans.append(v)
            v = []
    ans_ = []
    for i  in ans:
        if not i:
            pass
        else:
            ans_.append(i)

    return ans_


def main():
    p = readFile()
    print(p)
    print([len(i) for i in p])
if __name__ == '__main__':
    main()
