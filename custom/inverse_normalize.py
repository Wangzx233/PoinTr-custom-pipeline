import numpy as np
import open3d as o3d


def inverse_normalize_point_cloud(normalized_points, original_center, original_scale_factor):
    """
    对归一化的点云进行逆变换，恢复到原始坐标系
    
    Args:
        normalized_points (np.ndarray): 归一化后的点云 shape=(N, 3)
        original_center (np.ndarray): 原始点云的中心点 shape=(3,)
        original_scale_factor (float): 原始缩放因子
        
    Returns:
        np.ndarray: 恢复到原始坐标系的点云 shape=(N, 3)
    """
    # 在down_sample.py中的归一化逻辑是:
    # 1. 先平移到中心点: centered_points = points - center
    # 2. 再缩放: normalized_points = centered_points * scale_factor
    # 其中 scale_factor = 1.0 / max_extent

    print("处理后的缩放因子：", original_scale_factor)
    # 因此反归一化逻辑应该是:
    # 1. 逆缩放操作：乘以缩放因子的倒数(即max_extent)
    unscaled_points = normalized_points / original_scale_factor
    
    # 2. 逆平移操作：加回原始中心点
    original_points = unscaled_points + original_center
    
    return original_points


def Restore_point_cloud(pcd, original_center, original_scale_factor):
    """
    读取归一化的点云文件，恢复到原始坐标系并保存
    
    Args:
        pcd (o3d.geometry.PointCloud or np.ndarray): 输入点云（可以是Open3D点云对象或numpy数组）
        original_center (np.ndarray): 原始点云的中心点 shape=(3,)
        original_scale_factor (float): 原始缩放因子
    """
    # 判断输入类型并获取点云数据
    if isinstance(pcd, o3d.geometry.PointCloud):
        normalized_points = np.asarray(pcd.points)
        has_colors = pcd.has_colors()
        if has_colors:
            colors = pcd.colors
    else:
        # 假设输入是numpy数组
        normalized_points = pcd
        has_colors = False
    
    # 执行逆变换
    original_points = inverse_normalize_point_cloud(normalized_points, original_center, original_scale_factor)
    
    # 创建还原后的点云对象
    restored_pcd = o3d.geometry.PointCloud()
    restored_pcd.points = o3d.utility.Vector3dVector(original_points)
    
    # 如果原点云有颜色，保留颜色
    if has_colors:
        restored_pcd.colors = colors
    
    # 保存还原后的点云
    # o3d.io.write_point_cloud(output_file, restored_pcd)
    
    print(f"点云点数: {len(restored_pcd.points)}")
    
    return restored_pcd


if __name__ == "__main__":
    # 示例使用
    input_file = "normalized_point_cloud.pcd"  # 归一化后的点云文件
    output_file = "restored_point_cloud.pcd"   # 恢复后的点云文件
    
    # 这些参数需要在归一化时保存
    original_center = np.array([1.5, 2.0, 3.0])  # 示例值，实际使用时需替换为真实的中心点
    original_scale_factor = 4.0                  # 示例值，实际使用时需替换为真实的缩放因子
    
    # 执行恢复
    # restored_pcd = restore_point_cloud(input_file, output_file, original_center, original_scale_factor)