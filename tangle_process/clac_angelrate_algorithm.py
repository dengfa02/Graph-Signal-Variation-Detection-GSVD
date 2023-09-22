import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def cal_window_wtr(data, longitude, latitude, time_data, L_max, L_min, para_c):
    angles = []
    time_diffs_ave = []
    L_list = []
    for i in range(len(data) - 2):
        x1 = longitude[i]
        y1 = latitude[i]
        x2 = longitude[i + 1]
        y2 = latitude[i + 1]
        x3 = longitude[i + 2]
        y3 = latitude[i + 2]
        L1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        L2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        L = math.sqrt(L1 * L2)  # geometrical mean
        L_list.append(L)
        value = ((x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2)) / (L1 * L2 + 1e-10)
        if -1 <= value <= 1:
            value = value
        if value > 1:
            value = 1
        if value < -1:
            value = -1
        angle = math.acos(value) * 180.0 / math.pi
        angles.append(angle)
        time_diff1 = time_data[i + 1] - time_data[i]
        time_diff2 = time_data[i + 2] - time_data[i + 1]
        time_diff = max(time_diff1, time_diff2)
        time_diffs_ave.append(time_diff)

    delta_angles = [item_a / item_t for item_a, item_t in zip(angles, time_diffs_ave)]
    L_standard = [(item - L_min) / (L_max - L_min + 1e-10) for i, item in enumerate(L_list)]  # standardization
    WTR = np.array([item_d * item_l ** para_c for item_d, item_l in zip(delta_angles, L_standard)])
    WTR = np.nan_to_num(WTR)
    WTR = WTR.tolist()

    return WTR


def cal_wtr_algorithm(data, longitude, latitude, time_data, para_c):  # The default value for para_c is 1.47
    angles = []
    time_diffs_ave_1 = []
    L_list = []
    for i in range(len(data) - 2):
        x1 = longitude[i]
        y1 = latitude[i]
        x2 = longitude[i + 1]
        y2 = latitude[i + 1]
        x3 = longitude[i + 2]
        y3 = latitude[i + 2]
        L1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        L2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        L = math.sqrt(L1 * L2)  # geometrical mean
        L_list.append(L)
        value = ((x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2)) / (L1 * L2 + 1e-10)
        if -1 <= value <= 1:
            value = value
        if value > 1:
            value = 1
        if value < -1:
            value = -1
        angle = math.acos(value) * 180.0 / math.pi
        angles.append(angle)
        time_diff1 = time_data[i + 1] - time_data[i]
        time_diff2 = time_data[i + 2] - time_data[i + 1]
        time_diff_1 = max(time_diff1, time_diff2)
        time_diffs_ave_1.append(time_diff_1)

    L_max = max(L_list)
    L_min = min(L_list)
    delta_angles_1 = [item_a / (item_t + 1e-10) for item_a, item_t in zip(angles, time_diffs_ave_1)]
    L_standard = [(item - L_min) / (L_max - L_min + 1e-10) for i, item in enumerate(L_list)]  # standardization
    WTR1 = [item_d * item_l ** para_c for item_d, item_l in zip(delta_angles_1, L_standard)]

    return WTR1, L_max, L_min
