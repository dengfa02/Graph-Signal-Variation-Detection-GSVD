import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from slice_delete_same import slice_delete_same
from clac_angelrate_algorithm import cal_wtr_algorithm
from graphic_network_construct import compute_total_variation
from NN_disentangle import find_disentangle_path, vs_disentangle_plot
import time
from sklearn.cluster import KMeans
import math

"""
For the operation of the final method, it is recommended to load the data as the trajectory after segmentation.
"""


def harversin_distance_2points(latA, lonA, latB, lonB):
    r = 6371  # 地球半径,km
    latA = latA * math.pi / 180
    lonA = lonA * math.pi / 180
    latB = latB * math.pi / 180
    lonB = lonB * math.pi / 180
    if latA == latB and lonA == lonB:
        distance_nm = 0
    else:
        distance_km = 2 * r * math.asin(math.sqrt(
            math.sin((latB - latA) / 2) ** 2 + math.cos(latA) * math.cos(latB) * math.sin((lonB - lonA) / 2) ** 2))
        distance_nm = distance_km / 1.852  # 转换成海里
    return distance_nm


def total_algorithm(trace_data, id_index, speed_index, lon_index, lat_index, course_index, time_index, save_path,
                    window_size, step, para_c=1.47,
                    para_std=0.02):  # The default value: para_c is 1.47, para_std is 0.02
    # Remove duplicate traj points
    concat_5slicetrace_delete_same = slice_delete_same(trace_data, lon_index, lat_index)
    concat_5slicetrace_delete_same = np.array(concat_5slicetrace_delete_same)
    print("Duplicate trace points were removed successfully!")

    # remove drift points
    dis_list = []
    timediff_list = []
    speed_cal_list = []
    traj = []
    for i in range(len(concat_5slicetrace_delete_same) - 1):
        dis_list.append(harversin_distance_2points(concat_5slicetrace_delete_same[i, lat_index],
                                                   concat_5slicetrace_delete_same[i, lon_index],
                                                   concat_5slicetrace_delete_same[i + 1, lat_index],
                                                   concat_5slicetrace_delete_same[i + 1, lon_index]))
        timediff_list.append((concat_5slicetrace_delete_same[i + 1, time_index] -
                              concat_5slicetrace_delete_same[i, time_index]) / 3600)
        speed_cal_list.append(dis_list[i] / timediff_list[i])
    for i in range(len(speed_cal_list)):
        if speed_cal_list[i] <= 30:
            traj.append(concat_5slicetrace_delete_same[i, :])
        if i == len(speed_cal_list) - 1:
            traj.append(concat_5slicetrace_delete_same[-1, :])
    traj = np.vstack(traj)
    print("The drift points were removed successfully!")

    algorithm_data = traj
    longitude = algorithm_data[:, lon_index]
    latitude = algorithm_data[:, lat_index]
    time_data = algorithm_data[:, time_index]

    recog_start = time.time()
    DMF, L_max, L_min = cal_wtr_algorithm(algorithm_data, longitude, latitude, time_data, para_c)
    DMF = [0] + DMF + [0]  # DMF starts and ends at 0
    nan_index = np.argwhere(np.isnan(DMF)).flatten()
    DMF = np.array(DMF)
    DMF[nan_index] = 0
    DMF = DMF.tolist()

    traj_tuple = [tuple(x) for x in algorithm_data[:, (lon_index, lat_index)]]
    total_variation_array, WTR_list = compute_total_variation(algorithm_data, traj_tuple, DMF, window_size, step)

    if np.std(total_variation_array) <= para_std:
        q1, q3 = np.percentile(total_variation_array, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + (3 * iqr)
        outliers_index = np.array(np.where(total_variation_array > upper_bound))
        outliers_index = outliers_index.flatten()

    else:
        X = [[np.log10(x)] for x in total_variation_array]
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        labels = kmeans.labels_
        labels_0 = np.where(labels == 0)
        labels_1 = np.where(labels != 0)
        if len(labels_0[0]) < len(labels_1[0]):
            labels_out = labels_0
        else:
            labels_out = labels_1
        outliers_index = np.array(labels_out).flatten()

    recog_end = time.time()
    print("recognition time：{}ms".format((recog_end - recog_start) * 1000))

    ab_ids = [index_s * step for i, index_s in enumerate(outliers_index)]
    if len(ab_ids) != 0:
        start = ab_ids[0]
        start_list = []
        end_list = []
        for k, ids in enumerate(ab_ids):
            end = ab_ids[k] + window_size
            if k != len(ab_ids) - 1 and ab_ids[k + 1] - ab_ids[k] <= window_size:
                end = ab_ids[k + 1] + window_size
            else:
                start_list.append(start)
                end_list.append(end)
                start = ab_ids[k + 1] if k != len(ab_ids) - 1 else ab_ids[k]

        path = []  # Storage seeking path
        reconstruct_start = time.time()
        for i in range(len(start_list)):
            tangle_traj_tuple = [tuple(x) for x in
                                 algorithm_data[start_list[i]:end_list[i] + 1, (lon_index, lat_index)]]
            tangle_total = np.array([point for point in algorithm_data[start_list[i]:end_list[i] + 1, :]])
            path_in = find_disentangle_path(tangle_traj_tuple, tangle_total, lon_index, lat_index, time_index, L_max,
                                            L_min, para_c)
            path.append(path_in)
        reconstruct_end = time.time()
        print("reconstruction time：{}ms".format((reconstruct_end - reconstruct_start) * 1000))

        vs_disentangle_plot(start_list, end_list, algorithm_data, path, lon_index, lat_index, speed_index, course_index,
                            save_path=save_path)  # You can plot any parameter you care about in this function

        return start_list, end_list

    else:
        np.save(save_path, algorithm_data)
        return [], []


if __name__ == '__main__':

    mmsi = [245539000, 410050325]  # 2 trajs for example
    for m, item in enumerate(mmsi):
        try:
            trace_data = pd.read_csv(f'../dataset/{mmsi[m]}_ori.csv', encoding='gbk',
                                     delimiter=',')
        except UnicodeDecodeError:
            trace_data = pd.read_csv(f'../dataset/{mmsi[m]}_ori.csv', encoding='utf-8',
                                     delimiter=',')
        print("MMSI:", item)
        # trace_data = trace_data.sort_values(by=['UnixTime'], ascending=True)
        trace_data = trace_data.loc[:, ['id', 'Speed', 'Course', 'Lon_d', 'Lat_d', 'UnixTime']]
        trace_data = trace_data.dropna(axis=0, how='any')
        trace_data = trace_data[~np.isinf(trace_data).any(1) & ~np.isnan(trace_data).any(1)]
        trace_data = np.array(trace_data)

        id_index = 0
        lon_index = 3
        lat_index = 4
        time_index = 5
        speed_index = 1
        course_index = 2

        """
        parameter setting
        """
        window_size = 20
        step = 3
        para_c = 1.47
        para_std = 0.02
        save_path = f'../dataset/{mmsi[m]}_disentangle.npy'

        start_list, end_list = total_algorithm(trace_data, id_index, speed_index, lon_index, lat_index,
                                               course_index, time_index, save_path, window_size, step,
                                               para_c, para_std)
