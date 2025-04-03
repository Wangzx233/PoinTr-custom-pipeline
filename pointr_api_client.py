import requests
import os
import time
from typing import Optional, Dict, Any, Union, List

def complete_point_cloud(
    input_path: str,
    output_path: Optional[str] = None,
    server_url: str = "http://223.109.239.8:4011",
    target_points: int = 4096,
    sampling_method: str = "fps",
    timeout: int = 300,
    verbose: bool = True
) -> Dict[str, Any]:
    """调用点云补全API服务处理单个点云文件
    
    Args:
        input_path (str): 服务器上点云文件的绝对路径
        output_path (str, optional): 服务器上输出文件的绝对路径。如果不指定，将自动生成
        server_url (str): API服务器URL，默认 "http://223.109.239.8:4011"
        target_points (int): 采样点数，默认 4096
        sampling_method (str): 采样方法，可选 "fps", "random", "voxel"，默认 "fps"
        timeout (int): 请求超时时间（秒），默认 300
        verbose (bool): 是否显示详细信息，默认 True
        
    Returns:
        Dict[str, Any]: 包含处理结果的字典，至少包含 'status' 和 'output_file' 字段
        
    Raises:
        Exception: 当API请求失败或超时时抛出
    """
    # 验证输入路径
    if not os.path.isabs(input_path):
        raise ValueError(f"输入路径必须是绝对路径: {input_path}")
    
    # 如果未指定输出路径，则根据输入路径生成
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(input_name)[0]
        output_path = os.path.join(input_dir, f"{name_without_ext}_completed.ply")
    
    # 验证采样方法
    if sampling_method not in ["fps", "random", "voxel"]:
        raise ValueError(f"不支持的采样方法: {sampling_method}，可选 'fps', 'random', 'voxel'")
    
    # 准备请求数据
    request_data = {
        "input_file": input_path,
        "output_file": output_path,
        "target_points": target_points,
        "sampling_method": sampling_method
    }
    
    # 检查服务器健康状态
    if verbose:
        print(f"检查服务器 {server_url} 状态...")
    
    try:
        health_response = requests.get(f"{server_url}/health", timeout=10)
        health_data = health_response.json()
        
        if health_response.status_code != 200 or health_data.get("status") != "healthy":
            raise ConnectionError(f"服务器状态异常: {health_data}")
            
        if verbose:
            print("服务器正常运行")
    except Exception as e:
        raise ConnectionError(f"无法连接到服务器 {server_url}: {str(e)}")
    
    # 发送点云补全请求
    if verbose:
        print(f"发送点云补全请求: {input_path} → {output_path}")
        print(f"参数: 点数={target_points}, 采样方法={sampling_method}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{server_url}/complete_file",
            json=request_data,
            timeout=timeout
        )
        
        # 解析响应
        if response.status_code == 200:
            result = response.json()
            elapsed_time = time.time() - start_time
            
            if verbose:
                print(f"处理成功! 用时: {elapsed_time:.2f}秒")
                print(f"输出文件: {result['output_file']}")
            
            return {
                "status": "success",
                "input_file": result["input_file"],
                "output_file": result["output_file"],
                "elapsed_time": elapsed_time
            }
        else:
            error_data = response.json()
            error_message = error_data.get("detail", "未知错误")
            
            if verbose:
                print(f"处理失败: {error_message}")
            
            return {
                "status": "error",
                "error": error_message,
                "input_file": input_path
            }
    except requests.Timeout:
        if verbose:
            print(f"请求超时 (>{timeout}秒)")
        
        return {
            "status": "error",
            "error": f"请求超时 (>{timeout}秒)",
            "input_file": input_path
        }
    except Exception as e:
        if verbose:
            print(f"请求异常: {str(e)}")
        
        return {
            "status": "error",
            "error": str(e),
            "input_file": input_path
        }


def complete_point_cloud_folder(
    input_folder: str,
    output_folder: str,
    server_url: str = "http://223.109.239.8:4011",
    target_points: int = 4096,
    sampling_method: str = "fps",
    file_extension: str = ".ply",
    timeout: int = 600,
    verbose: bool = True,
    skip_files: List[str] = None
) -> Dict[str, Any]:
    """调用点云补全API服务处理整个文件夹的点云文件
    
    Args:
        input_folder (str): 服务器上输入文件夹的绝对路径
        output_folder (str): 服务器上输出文件夹的绝对路径
        server_url (str): API服务器URL，默认 "http://223.109.239.8:4011"
        target_points (int): 采样点数，默认 4096
        sampling_method (str): 采样方法，可选 "fps", "random", "voxel"，默认 "fps"
        file_extension (str): 处理的文件扩展名，默认 ".ply"
        timeout (int): 请求超时时间（秒），默认 600
        verbose (bool): 是否显示详细信息，默认 True
        skip_files (List[str], optional): 需要跳过补全直接复制的文件名列表（不含扩展名）
        
    Returns:
        Dict[str, Any]: 包含处理结果的字典，至少包含 'status', 'total_files', 'successful' 和 'results' 字段
        
    Raises:
        Exception: 当API请求失败或超时时抛出
    """
    # 验证输入路径
    if not os.path.isabs(input_folder):
        raise ValueError(f"输入文件夹路径必须是绝对路径: {input_folder}")
    
    if not os.path.isabs(output_folder):
        raise ValueError(f"输出文件夹路径必须是绝对路径: {output_folder}")
    
    # 验证采样方法
    if sampling_method not in ["fps", "random", "voxel"]:
        raise ValueError(f"不支持的采样方法: {sampling_method}，可选 'fps', 'random', 'voxel'")
    
    # 确保skip_files是一个列表
    skip_files = skip_files or []
    
    # 准备请求数据
    request_data = {
        "input_folder": input_folder,
        "output_folder": output_folder,
        "target_points": target_points,
        "sampling_method": sampling_method,
        "file_extension": file_extension,
        "skip_files": skip_files
    }
    
    # 检查服务器健康状态
    if verbose:
        print(f"检查服务器 {server_url} 状态...")
    
    try:
        health_response = requests.get(f"{server_url}/health", timeout=10)
        health_data = health_response.json()
        
        if health_response.status_code != 200 or health_data.get("status") != "healthy":
            raise ConnectionError(f"服务器状态异常: {health_data}")
            
        if verbose:
            print("服务器正常运行")
    except Exception as e:
        raise ConnectionError(f"无法连接到服务器 {server_url}: {str(e)}")
    
    # 发送点云补全请求
    if verbose:
        print(f"发送文件夹处理请求: {input_folder} → {output_folder}")
        print(f"参数: 点数={target_points}, 采样方法={sampling_method}, 文件扩展名={file_extension}")
        if skip_files:
            print(f"跳过补全的文件: {len(skip_files)} 个")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{server_url}/complete_folder",
            json=request_data,
            timeout=timeout
        )
        
        # 解析响应
        if response.status_code == 200:
            result = response.json()
            elapsed_time = time.time() - start_time
            
            if verbose:
                print(f"处理成功! 用时: {elapsed_time:.2f}秒")
                print(f"总文件数: {result['total_files']}, 成功处理: {result['successful']}")
                if 'copied_files' in result:
                    print(f"直接复制: {result['copied_files']}个, 点云补全: {result['completed_files']}个")
            
            return {
                "status": "success",
                "input_folder": input_folder,
                "output_folder": output_folder,
                "total_files": result["total_files"],
                "successful": result["successful"],
                "copied_files": result.get("copied_files", 0),
                "completed_files": result.get("completed_files", result["successful"]),
                "results": result["results"],
                "elapsed_time": elapsed_time
            }
        else:
            error_data = response.json()
            error_message = error_data.get("detail", "未知错误")
            
            if verbose:
                print(f"处理失败: {error_message}")
            
            return {
                "status": "error",
                "error": error_message,
                "input_folder": input_folder,
                "output_folder": output_folder
            }
    except requests.Timeout:
        if verbose:
            print(f"请求超时 (>{timeout}秒)")
        
        return {
            "status": "error",
            "error": f"请求超时 (>{timeout}秒)",
            "input_folder": input_folder,
            "output_folder": output_folder
        }
    except Exception as e:
        if verbose:
            print(f"请求异常: {str(e)}")
        
        return {
            "status": "error",
            "error": str(e),
            "input_folder": input_folder,
            "output_folder": output_folder
        }


# 使用示例
if __name__ == "__main__":
    # # 单文件示例
    # result = complete_point_cloud(
    #     input_path="/path/to/input.ply",
    #     output_path="/path/to/output.ply",
    #     target_points=4096,
    #     sampling_method="fps"
    # )
    #
    # if result["status"] == "success":
    #     print(f"点云补全成功，输出文件: {result['output_file']}")
    # else:
    #     print(f"点云补全失败: {result['error']}")
    
    # 文件夹示例 - 带跳过列表
    skip_file_list = ["file1", "file2"]  # 这些文件将被直接复制而不进行补全处理
    
    folder_result = complete_point_cloud_folder(
        input_folder="/home/wangxin/api/PoinTr-custom-pipeline/rotated_2",
        output_folder="/home/wangxin/api/PoinTr-custom-pipeline/rotated_2_output",
        target_points=8192,
        sampling_method="fps",
        file_extension=".ply",
        skip_files=skip_file_list
    )

    print("输出文件夹：", folder_result["output_folder"])
    if folder_result["status"] == "success":
        print(f"文件夹处理成功，共 {folder_result['successful']}/{folder_result['total_files']} 个文件")
        
        # 分别统计复制和补全的文件数
        copied_count = len([r for r in folder_result["results"] if r["status"] == "copied"])
        completed_count = len([r for r in folder_result["results"] if r["status"] == "success"])
        failed_count = len([r for r in folder_result["results"] if r["status"] not in ["copied", "success"]])
        
        print(f"- 直接复制: {copied_count} 个文件")
        print(f"- 点云补全: {completed_count} 个文件")
        print(f"- 处理失败: {failed_count} 个文件")
    else:
        print(f"文件夹处理失败: {folder_result['error']}") 