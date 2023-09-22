import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from clac_angelrate_algorithm import cal_window_wtr


def compute_window_variation(tangle_total, path_op, lon_index, lat_index, time_index, traj_tuple, L_max, L_min, para_c):
    """
    Calculate window variation
    """
    disentangle_data = []
    for j, item in enumerate(path_op):
        disentangle_data.append(tangle_total[item, (lon_index, lat_index)])
    disentangle_data = np.vstack(disentangle_data)
    longitude = disentangle_data[:, 0]
    latitude = disentangle_data[:, 1]
    time_data = tangle_total[:, time_index]

    dmf = cal_window_wtr(disentangle_data, longitude, latitude, time_data, L_max, L_min, para_c)
    dmf = [0] + dmf + [0]

    graph_window = nx.Graph()
    for i, node in enumerate(traj_tuple):
        graph_window.add_node(i, pos=node)
        if i < len(traj_tuple) - 1:
            graph_window.add_edge(i, i + 1)
    for j, indicator in enumerate(dmf):
        graph_window.nodes[j]['wtr'] = indicator

    L_win = nx.laplacian_matrix(graph_window).todense()
    DMF = np.array([graph_window.nodes[y]['wtr'] for y in graph_window.nodes()])
    window_variation = np.matmul(np.matmul(DMF.T, L_win), DMF)

    return window_variation


def compute_total_variation(data_net, traj_tuple, wtrs, window_size, step):
    total_variation_list = []
    WTR_list = []
    win_index = 0

    for n in range(0, len(data_net) - window_size, step):
        graph = nx.Graph()
        for i, node in enumerate(traj_tuple[n:n + window_size]):
            graph.add_node(i + n, pos=node)
            if i < len(traj_tuple[n:n + window_size]) - 1:
                graph.add_edge(i + n, i + n + 1)
        for j, indicator in enumerate(wtrs[n:n + window_size]):
            graph.nodes[j + n]['wtr'] = indicator

        L = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 2, -1, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, 2, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 2, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 2, -1, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1, 2, -1, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
                      [0, 0, 0, 0, 0, 0, 0, 0, -1, 1]])
        WTR = np.array([graph.nodes[i]['wtr'] for i in graph.nodes()])
        total_variation = np.matmul(np.matmul(WTR.T, L), WTR)
        total_variation_list.append(total_variation)
        WTR_list.append(WTR)
        win_index += 1
    graph_tail = nx.Graph()
    for k, node in enumerate(traj_tuple[(len(data_net) - window_size) - (len(data_net) - window_size) % step:]):
        graph_tail.add_node(k, pos=node)
        if k < len(traj_tuple[(len(data_net) - window_size) - (
                len(data_net) - window_size) % step:]) - 1:
            graph_tail.add_edge(k, k + 1)
    for j, indicator in enumerate(wtrs[(len(data_net) - window_size) - (len(data_net) - window_size) % step:]):
        graph_tail.nodes[j]['wtr'] = indicator

    L_tail = nx.laplacian_matrix(graph_tail).todense()
    WTR = np.array([graph_tail.nodes[j]['wtr'] for j in graph_tail.nodes()])
    total_variation = np.sum(np.matmul(np.matmul(WTR.T, L_tail), WTR))
    total_variation_list.append(total_variation)
    WTR_list.append(WTR)
    win_index += 1

    return np.array(total_variation_list), WTR_list
