import argparse
import numpy as np
import open3d as o3d
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from custom.down_sample import Process_point_cloud
from custom.inference import Inference
from custom.inverse_normalize import Restore_point_cloud
normal_record_map = {}

def batch_process_point_clouds(input_dir, output_dir, target_points=2048, sampling_method='fps', file_extension='.ply',args=None):
    """批量处理文件夹中的点云文件

    Args:
        input_dir (str): 输入点云文件夹路径
        output_dir (str): 输出点云文件夹路径
        target_points (int): 采样后的点数
        sampling_method (str): 采样方法, 'random'、'fps'或'voxel'
        file_extension (str): 点云文件扩展名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有指定扩展名的文件
    ply_files = [f for f in os.listdir(input_dir) if f.lower().endswith(file_extension)]

    if not ply_files:
        print(f"在 {input_dir} 中没有找到 {file_extension} 文件")
        return

    print(f"找到 {len(ply_files)} 个{file_extension}文件，开始处理...")

    # 处理每个文件
    success_count = 0
    for filename in tqdm(ply_files):
        input_path = os.path.join(input_dir, filename)

        # 保持相同的文件名但更改扩展名为.ply
        output_filename = os.path.splitext(filename)[0] + '.ply'
        output_path = os.path.join(output_dir, output_filename)

        try:
            success,center,scale_factor,pcd_filtered = Process_point_cloud(input_path, target_points, sampling_method)
            # 将归一化参数保存到字典中
            normal_record_map[output_filename] = (center, scale_factor)
            if success:
                success_count += 1
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            continue

        pcd_out = Inference(pcd_filtered,args)

        # 从字典中获取当前文件的原始中心点和缩放因子
        current_center, current_scale_factor = normal_record_map[output_filename]
        restored_pcd = Restore_point_cloud(pcd_out, current_center, current_scale_factor)

        # 统计离群点移除
        cl, ind = restored_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        resuilt_pcd = restored_pcd.select_by_index(ind)
        # print("移除离群点后点云点数：", len(pcd_filtered.points))

        o3d.io.write_point_cloud(output_path, resuilt_pcd)

    # print(f"成功正则化，采样,移除离群点 {success_count}/{len(ply_files)} 个文件")
    print(f"成功处理 {success_count}/{len(ply_files)} 个文件")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config',
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint',
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud')
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

if __name__ == "__main__":
    # 使用示例
    input_directory = "rotated_2"  # 包含.ply文件的输入目录
    output_directory = "rotated_2_out"  # 处理后输出的目录

    args = get_args()
    # 调用批处理函数
    batch_process_point_clouds(
        input_dir=input_directory,
        output_dir=output_directory,
        target_points=8192,  # PoinTr模型需要2048个点
        sampling_method='fps',  # 'random', 'fps', 或 'voxel'
        file_extension='.ply',
        args=args
    )