import torch
import pandas as pd
import numpy as np
from haversine import haversine,Unit
from torch_geometric.utils import dense_to_sparse
import math

def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x, y)  # 计算初始方位角，以弧度值表示
    brng = math.degrees(brng)  # 转化为度
    brng = (brng + 360) % 360  # 确保结果在0-360度范围内
    return brng


def get_adjacency_matrix():
    station_df = pd.read_csv('Station_info/Station_bj.csv', encoding='gb18030')
    sensor_ids = station_df['监测点ID']
    num_sensors = len(station_df)

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # 初始化坐标矩阵 [num_sensors, 2] -> 每一行是[纬度, 经度]
    node_pos = np.zeros((num_sensors, 2))
    for i in range(num_sensors):
        node_pos[i, 0] = station_df.loc[i, '纬度']
        node_pos[i, 1] = station_df.loc[i, '经度']

    # 计算邻接矩阵
    dist = np.zeros((num_sensors, num_sensors))  # 距离
    dist_mx = np.zeros((num_sensors, num_sensors))  # 距离权重矩阵
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            coords1 = station_df.loc[i, '纬度'], station_df.loc[i, '经度']
            coords2 = station_df.loc[j, '纬度'], station_df.loc[j, '经度']
            distance = haversine(coords1, coords2, unit=Unit.KILOMETERS)
            if distance <= 60:
                dist_mx[i, j] = 1 / distance
                dist_mx[j, i] = 1 / distance
                dist[i, j] = distance
                dist[j, i] = distance
    adj_mx = dist_mx.copy()
    adj_dist = dist.copy()

    edge_index, dist_diff = dense_to_sparse(torch.tensor(adj_mx))
    edge_index, dist_adv = dense_to_sparse(torch.tensor(adj_dist))
    edge_index, dist_adv, dist_diff = edge_index.numpy(), dist_adv.numpy(), dist_diff.numpy()

    dist_diff_arr = []
    dist_adv_arr = []
    direc_arr = []

    for i in range(edge_index.shape[1]):  # 有几条边0->1,1->2
        src, dest = edge_index[0, i], edge_index[1, i]
        src_lat, src_lon = station_df['纬度'][src], station_df['经度'][src]
        dest_lat, dest_lon = station_df['纬度'][dest], station_df['经度'][dest]
        src_location = (src_lat, src_lon)
        dest_location = (dest_lat, dest_lon)

        # 计算方位角
        dist_diff_km = dist_diff[i,]
        dist_adv_km = dist_adv[i,]
        bearing = get_bearing(src_lat, src_lon, dest_lat, dest_lon)

        dist_diff_arr.append(dist_diff_km)
        dist_adv_arr.append(dist_adv_km)
        direc_arr.append(bearing)

    dist_diff_arr = np.stack(dist_diff_arr)
    dist_adv_arr = np.stack(dist_adv_arr)
    direc_arr = np.stack(direc_arr)
    edge_attr = np.stack([dist_diff_arr, dist_adv_arr, direc_arr], axis=-1)  # 距离和方位角

    return adj_mx, edge_index, edge_attr, node_pos  # 新增返回 node_pos

