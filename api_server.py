from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os
import numpy as np
import open3d as o3d
import tempfile
import argparse
import uuid
import base64
import uvicorn
from typing import Optional
from pydantic import BaseModel
from pipeline import Process_point_cloud, Inference, Restore_point_cloud
from io import BytesIO

app = FastAPI()

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
        help='Device used for inference')
    parser.add_argument(
        '--port',
        type=int,
        default=4011,
        help='Port to run the API server on')
    args = parser.parse_args()
    return args

class FolderProcessRequest(BaseModel):
    input_folder: str
    output_folder: str
    target_points: int = 8192
    sampling_method: str = 'fps'
    file_extension: str = '.ply'

class FileProcessRequest(BaseModel):
    input_file: str
    output_file: str
    target_points: int = 8192
    sampling_method: str = 'fps'

@app.get('/health')
def health_check():
    return {"status": "healthy"}

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
            
        except Exception as e:
            results.append({
                "file": filename,
                "status": "failed",
                "error": str(e)
            })
    
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
        
        return {
            "status": "success",
            "input_file": request.input_file,
            "output_file": request.output_file
        }
        
    except Exception as e:
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
    
    # Store args in app state
    app.state.args = args
    
    print(f"Starting API server on port {args.port}...")
    print(f"Model config: {args.model_config}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Device: {args.device}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    start() 