from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import open3d as o3d
import tempfile
import argparse
import uuid
import base64
import uvicorn
import torch
import gc
from typing import Optional, List
from pydantic import BaseModel
from pipeline import Process_point_cloud, Inference, Restore_point_cloud
from io import BytesIO

app = FastAPI()

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# Global variable to store normalization parameters
normal_record_map = {}

# Parse command line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_config',
        required=True,
        help='yaml config file')
    parser.add_argument(
        '--model_checkpoint',
        required=True,
        help='pretrained weight')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='Device used for inference, e.g. cuda:0, cuda:1, etc.')
    parser.add_argument(
        '--auto_select_device',
        action='store_true',
        help='Automatically select the CUDA device with the most free memory')
    parser.add_argument(
        '--port',
        type=int,
        default=4011,
        help='Port to run the API server on')
    parser.add_argument(
        '--max_memory',
        type=float,
        default=0.7,
        help='Maximum GPU memory fraction to use (0.0-1.0)')
    args = parser.parse_args()
    return args

class FolderProcessRequest(BaseModel):
    input_folder: str
    output_folder: str
    target_points: int = 4096  # 降低默认点数
    sampling_method: str = 'fps'
    file_extension: str = '.ply'

class FileProcessRequest(BaseModel):
    input_file: str
    output_file: str
    target_points: int = 4096  # 降低默认点数
    sampling_method: str = 'fps'

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_info() -> dict:
    """获取所有GPU的内存信息"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        info = {}
        for i in range(device_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - allocated_memory
            
            info[f"cuda:{i}"] = {
                "total": total_memory / (1024**3),  # GB
                "allocated": allocated_memory / (1024**3),  # GB
                "reserved": reserved_memory / (1024**3),  # GB
                "free": free_memory / (1024**3)  # GB
            }
        return info
    else:
        return {"error": "CUDA not available"}

def find_best_device() -> str:
    """自动选择内存最多的GPU设备"""
    if not torch.cuda.is_available():
        return "cpu"
    
    gpu_info = get_gpu_info()
    if "error" in gpu_info:
        return "cpu"
    
    # 按可用内存降序排序
    devices = sorted(gpu_info.keys(), key=lambda x: gpu_info[x]["free"], reverse=True)
    if not devices:
        return "cpu"
    
    return devices[0]  # 返回可用内存最多的设备

def validate_device(device: str) -> tuple:
    """验证设备是否可用，返回 (是否可用, 错误信息)"""
    if device == "cpu":
        return True, ""
    
    if not torch.cuda.is_available():
        return False, "CUDA 不可用，请使用 CPU 或检查 CUDA 安装"
    
    try:
        if ":" in device:
            device_id = int(device.split(":")[1])
        else:
            device_id = 0
        
        if device_id >= torch.cuda.device_count():
            available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            return False, f"设备 {device} 不存在。可用设备: {', '.join(available_devices)}"
        
        # 测试设备是否正常工作
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        return True, ""
    except Exception as e:
        return False, f"设备 {device} 验证失败: {str(e)}"

@app.get('/health')
def health_check():
    return {"status": "healthy"}

@app.get('/gpu_info')
def gpu_info_endpoint():
    """获取GPU内存使用情况"""
    return get_gpu_info()

@app.get('/devices')
def available_devices():
    """获取所有可用设备"""
    if not torch.cuda.is_available():
        return {"devices": ["cpu"], "recommended": "cpu"}
    
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    devices.append("cpu")
    
    best_device = find_best_device()
    
    return {
        "devices": devices,
        "recommended": best_device,
        "current": app.state.device
    }

@app.post('/complete_folder')
async def complete_folder(request: FolderProcessRequest):
    """
    Process all point cloud files in a folder and save results to output folder
    """
    # Validate input folder
    if not os.path.exists(request.input_folder):
        raise HTTPException(status_code=400, detail=f"Input folder '{request.input_folder}' does not exist")
    
    # Create output folder if it doesn't exist
    os.makedirs(request.output_folder, exist_ok=True)
    
    # Get all files with the specified extension
    files = [f for f in os.listdir(request.input_folder) if f.lower().endswith(request.file_extension)]
    
    if not files:
        raise HTTPException(status_code=400, detail=f"No {request.file_extension} files found in the input folder")
    
    results = []
    success_count = 0
    
    # Process each file
    for filename in files:
        input_path = os.path.join(request.input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.ply'
        output_path = os.path.join(request.output_folder, output_filename)
        
        try:
            # Process the point cloud
            success, center, scale_factor, pcd_filtered = Process_point_cloud(
                input_path, 
                request.target_points, 
                request.sampling_method
            )
            
            if not success:
                results.append({
                    "file": filename,
                    "status": "failed",
                    "error": "Failed to process point cloud"
                })
                continue
            
            # Store normalization parameters
            normal_record_map[output_filename] = (center, scale_factor)
            
            # Run inference
            pcd_out = Inference(pcd_filtered, app.state.args)
            
            # Restore the point cloud
            current_center, current_scale_factor = normal_record_map[output_filename]
            restored_pcd = Restore_point_cloud(pcd_out, current_center, current_scale_factor)
            
            # Remove statistical outliers
            cl, ind = restored_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
            result_pcd = restored_pcd.select_by_index(ind)
            
            # Save the result
            o3d.io.write_point_cloud(output_path, result_pcd)
            
            success_count += 1
            results.append({
                "file": filename,
                "status": "success",
                "output_path": output_path
            })
            
            # 清理GPU内存
            clear_gpu_memory()
            
        except Exception as e:
            results.append({
                "file": filename,
                "status": "failed",
                "error": str(e)
            })
            # 出错后也清理内存
            clear_gpu_memory()
    
    return {
        "total_files": len(files),
        "successful": success_count,
        "results": results
    }

@app.post('/complete_file')
async def complete_file(request: FileProcessRequest):
    """
    Process a single point cloud file and save the result to the specified output path
    """
    # Validate input file
    if not os.path.exists(request.input_file):
        raise HTTPException(status_code=400, detail=f"Input file '{request.input_file}' does not exist")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(request.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process the point cloud
        success, center, scale_factor, pcd_filtered = Process_point_cloud(
            request.input_file, 
            request.target_points, 
            request.sampling_method
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process point cloud")
        
        # Store normalization parameters
        output_filename = os.path.basename(request.output_file)
        normal_record_map[output_filename] = (center, scale_factor)
        
        # Run inference
        pcd_out = Inference(pcd_filtered, app.state.args)
        
        # Restore the point cloud
        current_center, current_scale_factor = normal_record_map[output_filename]
        restored_pcd = Restore_point_cloud(pcd_out, current_center, current_scale_factor)
        
        # Remove statistical outliers
        cl, ind = restored_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        result_pcd = restored_pcd.select_by_index(ind)
        
        # Save the result
        o3d.io.write_point_cloud(request.output_file, result_pcd)
        
        # 清理GPU内存
        clear_gpu_memory()
        
        return {
            "status": "success",
            "input_file": request.input_file,
            "output_file": request.output_file
        }
        
    except Exception as e:
        # 出错后也清理内存
        clear_gpu_memory()
        raise HTTPException(status_code=500, detail=str(e))

def start():
    args = get_args()
    
    # Validate required arguments
    if not os.path.exists(args.model_config):
        print(f"Error: Model config file {args.model_config} not found")
        exit(1)
    
    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Model checkpoint file {args.model_checkpoint} not found")
        exit(1)
    
    # 设置CUDA设备
    if args.auto_select_device:
        args.device = find_best_device()
        print(f"自动选择设备: {args.device}")
    
    # 验证设备是否可用
    is_valid, error_msg = validate_device(args.device)
    if not is_valid:
        print(f"错误: {error_msg}")
        print(f"将尝试自动选择最佳设备...")
        args.device = find_best_device()
        is_valid, error_msg = validate_device(args.device)
        if not is_valid:
            print(f"错误: 无法找到可用设备: {error_msg}")
            exit(1)
    
    print(f"\n可用设备信息:")
    gpu_info = get_gpu_info()
    for device, info in gpu_info.items():
        print(f"{device}: 总内存 {info['total']:.2f}GB, 已分配 {info['allocated']:.2f}GB, 可用 {info['free']:.2f}GB")
    
    # 设置最大内存使用率
    if torch.cuda.is_available() and args.device != "cpu":
        try:
            if ":" in args.device:
                device_id = int(args.device.split(":")[1])
            else:
                device_id = 0
                
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(args.max_memory, device_id)
            
            # 设置活跃设备
            torch.cuda.set_device(device_id)
            
            print(f"使用设备 {args.device}，内存限制为 {args.max_memory*100:.0f}%")
        except Exception as e:
            print(f"设置设备失败: {str(e)}")
            exit(1)
    
    # 存储设备信息到应用状态
    app.state.device = args.device
    app.state.args = args
    
    print(f"\n启动API服务器，端口: {args.port}...")
    print(f"模型配置: {args.model_config}")
    print(f"模型权重: {args.model_checkpoint}")
    print(f"使用设备: {args.device}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    start() 