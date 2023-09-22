import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
This section is divided into trajectory points to remove duplications,
and only the adjacent points with different latitude and longitude are retained.
"""


def slice_delete_same(data, col1, col2):
    uni_data = []
    for i in range(len(data) - 1):
        if (data[i, col1] != data[i + 1, col1] and data[i, col2] != data[i + 1, col2]) or (
                data[i, col1] == data[i + 1, col1] and data[i, col2] != data[i + 1, col2]) or (
                data[i, col1] != data[i + 1, col1] and data[i, col2] == data[i + 1, col2]):
            uni_data.append(data[i])
        if i + 1 == len(data) - 1:  # Determine whether to reach the last point, and add the last point
            uni_data.append(data[i + 1])
    if len(uni_data) == 1:
        return uni_data
    else:
        uni_data = np.vstack(uni_data)
        return uni_data


if __name__ == '__main__':
    data = pd.read_csv('xxx.csv', encoding='GBK')
    data = data.loc[:, ['id', 'Speed', 'Course', 'Lon_d', 'Lat_d', 'UnixTime']].values
    data_de = slice_delete_same(data, 3, 4)
    pd.DataFrame(data_de).to_csv('xxx.csv')
