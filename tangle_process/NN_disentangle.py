import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from graphic_network_construct import compute_window_variation


def find_disentangle_path(traj_tuple, tangle_total, lon_index, lat_index, time_index, L_max, L_min, para_c):
    """
    This method is a greedy algorithm,
    which takes the first point as the starting point and finds the next point closest to its Euclidean distance.
    The visited points are no longer traversed,
    and a visited Boolean array is created to record the visits of the points.
    -param traj_tuple: The two-dimensional coordinates of each point and each element are of a tuple type.
    -return: path: The index list of each nearest point obtained by greedy algorithm.
    """
    # forward search
    num_points = len(traj_tuple)
    visited_op = np.zeros(num_points, dtype=bool)
    visited_op[0] = True
    path_op = [0]
    while not all(visited_op):
        last_point_op = traj_tuple[path_op[-1]]
        distances_op = np.zeros(num_points)
        for i in range(num_points):
            if not visited_op[i]:
                distances_op[i] = distance.euclidean(last_point_op, traj_tuple[i])
            else:
                distances_op[i] = np.inf
        next_point_op = np.argmin(distances_op)
        path_op.append(next_point_op)
        visited_op[next_point_op] = True

    # backward search
    visited_ne = np.zeros(num_points, dtype=bool)
    visited_ne[-1] = True
    path_ne = [num_points - 1]
    while not all(visited_ne):
        last_point_ne = traj_tuple[path_ne[-1]]
        distances_ne = np.zeros(num_points)
        for i in range(num_points):
            if not visited_ne[i]:
                distances_ne[i] = distance.euclidean(last_point_ne, traj_tuple[i])
            else:
                distances_ne[i] = np.inf
        next_point_ne = np.argmin(distances_ne)
        path_ne.append(next_point_ne)
        visited_ne[next_point_ne] = True

    path_ne_reversed = path_ne[::-1]
    path_origin = [i for i in range(num_points)]

    tv_or = compute_window_variation(tangle_total, path_origin, lon_index, lat_index, time_index, traj_tuple, L_max,
                                     L_min, para_c)
    tv_op = compute_window_variation(tangle_total, path_op, lon_index, lat_index, time_index, traj_tuple, L_max,
                                     L_min, para_c)
    tv_ne = compute_window_variation(tangle_total, path_ne, lon_index, lat_index, time_index, traj_tuple, L_max,
                                     L_min, para_c)

    value_to_path = {tv_or: path_origin, tv_op: path_op, tv_ne: path_ne_reversed}
    min_value = min(tv_or, tv_op, tv_ne)
    min_path = value_to_path[min_value]
    return min_path


def vs_disentangle_plot(tangle_start_index, tangle_end_index, algorithm_data, path, lon_index, lat_index, speed_index,
                        course_index, save_path):
    """
    You can graph any parameter you care about in this function.
    The following section replaces the reconstructed trajectory with the original trajectory.
    """
    for k in range(len(tangle_start_index)):
        traj_data = [x for x in algorithm_data[tangle_start_index[k]:tangle_end_index[k] + 1, :]]
        traj_data = np.vstack(traj_data)
        disentangle_data = []

        for j, item in enumerate(path[k]):
            disentangle_data.append(traj_data[item, :])
        disentangle_data = np.vstack(disentangle_data)

        algorithm_data[tangle_start_index[k]:tangle_end_index[k] + 1,
        (lon_index, lat_index, speed_index, course_index)] = disentangle_data[:,
                                                             (lon_index, lat_index, speed_index, course_index)]
        np.save(save_path, algorithm_data)

    """your plot program here..."""
