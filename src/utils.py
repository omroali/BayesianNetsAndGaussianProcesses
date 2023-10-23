# import numpy as np
import pandas as pd


def readCsv(filePath, training=False):
    read_data = pd.read_csv(filePath)
    header_data = list(read_data.columns)
    reponse_data = {}
    headers = {}

    if training:
        headers['data'] = header_data[:-1]
        headers['result'] = header_data[-1]
        reponse_data['data'] = read_data.iloc[:, :-1]
        reponse_data['result'] = read_data.iloc[:, -1]  # i.e. the last column
        print(f'Training data formatted to evaluate {headers["result"]}')
    else:
        headers['data'] = header_data
        reponse_data['data'] = read_data
        print('Testing data formatted')

    return {
        'type': 'training' if training else 'testing',
        'headers': headers,
        'data': reponse_data,
    }


def histogram(data):
    # csv 
    # x0,x1,x2,x3,x4,x5
    #  c, l, o, s, e, _
    return 0


def getHeaders(data):
    return data[0].split(',')
