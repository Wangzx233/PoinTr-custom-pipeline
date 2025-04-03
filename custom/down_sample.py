import numpy as np
import open3d as o3d
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def random_sampling(points, n_points):
    """随机采样

    Args:
        points (np.ndarray): 输入点云 shape=(N, 3)
        n_points (int): 采样后的点数

    Returns:
        np.ndarray: 采样后的点云 shape=(n_points, 3)
    """
    # 如果点数不足，直接返回原始点云
    if points.shape[0] <= n_points:
        return points

    # 随机选择n_points个点
    indices = np.random.choice(points.shape[0], n_points, replace=False)
    return points[indices]


def farthest_point_sampling(points, n_points):
    """最远点采样 (FPS)

    Args:
        points (np.ndarray): 输入点云 shape=(N, 3)
        n_points (int): 采样后的点数

    Returns:
        np.ndarray: 采样后的点云 shape=(n_points, 3)
    """
    # 如果点数不足，直接返回原始点云
    if points.shape[0] <= n_points:
        return points

    # 初始化采样点索引列表和距离列表
    N = points.shape[0]
    selected_indices = np.zeros(n_points, dtype=np.int32)
    distances = np.ones(N) * 1e10

    # 随机选择第一个点
    farthest_idx = np.random.randint(0, N)

    # 迭代选择最远点
    for i in range(n_points):
        selected_indices[i] = farthest_idx
        centroid = points[farthest_idx].reshape(1, 3)

        # 计算剩余点到当前点的距离
        dist = np.sum((points - centroid) ** 2, axis=1)

        # 更新最短距离
        mask = dist < distances
        distances[mask] = dist[mask]

        # 选择距离最远的点作为下一个点
        farthest_idx = np.argmax(distances)

    return points[selected_indices]


def voxel_down_sampling(points, voxel_size):
    """体素下采样

    Args:
        points (np.ndarray): 输入点云 shape=(N, 3)
        voxel_size (float): 体素大小

    Returns:
        np.ndarray: 下采样后的点云
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 应用体素下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled_pcd.points)


def Process_point_cloud(input_file, target_points=2048, sampling_method='fps'):
    """处理单个点云文件

    Args:
        input_file (str): 输入点云文件路径
        output_file (str): 输出点云文件路径
        target_points (int): 采样后的点数
        sampling_method (str): 采样方法, 'random'、'fps'或'voxel'
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)

    # 检查点云是否为空
    if len(points) == 0:
        print(f"警告: {input_file} 是空点云，跳过处理")
        return False

    # 归一化点云
    # centroid = np.mean(points, axis=0)
    # points_centered = points - centroid
    # max_distance = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    # if max_distance == 0:
    #     print(f"警告: {input_file} 所有点重合，跳过处理")
    #     return False
    # points_normalized = points_centered / max_distance
        # 计算点云的边界框
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # 计算点云的范围
    extents = max_coords - min_coords

    # 找到最大的尺寸，保持比例一致
    max_extent = np.max(extents)

    if max_extent == 0:
        print("警告: 点云是平面或线性的，无法正常归一化")
        return points

        # 计算中心点
    center = (min_coords + max_coords) / 2

    # 平移点云到原点为中心
    centered_points = points - center

    # 缩放点云使最大尺寸为1.0（即范围为[-0.5, 0.5]）
    scale_factor = 1.0 / max_extent
    normalized_points = centered_points * scale_factor

    print("处理前的缩放因子：",scale_factor)
    # 根据选择的方法对点云进行采样
    if sampling_method == 'random':
        sampled_points = random_sampling(normalized_points, target_points)
    elif sampling_method == 'fps':
        sampled_points = farthest_point_sampling(normalized_points, target_points)
    elif sampling_method == 'voxel':
        # 使用体素下采样后再用FPS精确控制点数
        voxel_size = 0.02  # 可以根据点云特性调整
        downsampled = voxel_down_sampling(normalized_points, voxel_size)
        sampled_points = farthest_point_sampling(downsampled, target_points)
    else:
        raise ValueError(f"不支持的采样方法: {sampling_method}")

    # 创建新的点云对象
    pcd_processed = o3d.geometry.PointCloud()
    pcd_processed.points = o3d.utility.Vector3dVector(sampled_points)

    # 统计离群点移除
    cl, ind = pcd_processed.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    pcd_filtered = pcd_processed.select_by_index(ind)
    print("移除离群点后点云点数：", len(pcd_filtered.points))
    # 保存处理后的点云
    # o3d.io.write_point_cloud(output_file, pcd_processed)
    return True,center,scale_factor,pcd_filtered
