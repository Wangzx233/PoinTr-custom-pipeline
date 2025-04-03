from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import open3d as o3d
import tempfile
import argparse
import uuid
import base64
from pipeline import Process_point_cloud, Inference, Restore_point_cloud
from io import BytesIO

app = Flask(__name__)

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
        default=5000,
        help='Port to run the API server on')
    args = parser.parse_args()
    return args

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/complete', methods=['POST'])
def complete_point_cloud():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Get parameters from request
    target_points = int(request.form.get('target_points', 8192))
    sampling_method = request.form.get('sampling_method', 'fps')
    
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    file.save(input_path)
    
    # Generate a unique output filename
    output_filename = f"{uuid.uuid4()}.ply"
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        # Process the point cloud
        success, center, scale_factor, pcd_filtered = Process_point_cloud(
            input_path, 
            target_points, 
            sampling_method
        )
        
        if not success:
            return jsonify({"error": "Failed to process point cloud"}), 500
        
        # Store normalization parameters
        normal_record_map[output_filename] = (center, scale_factor)
        
        # Run inference
        pcd_out = Inference(pcd_filtered, app.config['args'])
        
        # Restore the point cloud
        current_center, current_scale_factor = normal_record_map[output_filename]
        restored_pcd = Restore_point_cloud(pcd_out, current_center, current_scale_factor)
        
        # Remove statistical outliers
        cl, ind = restored_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        result_pcd = restored_pcd.select_by_index(ind)
        
        # Save the result
        o3d.io.write_point_cloud(output_path, result_pcd)
        
        # Return options
        response_format = request.form.get('response_format', 'file')
        
        if response_format == 'file':
            # Return the file
            return send_file(output_path, 
                            as_attachment=True,
                            download_name=output_filename)
        elif response_format == 'base64':
            # Convert to base64
            with open(output_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({
                "completion_successful": True,
                "point_cloud_base64": encoded
            })
        else:
            return jsonify({"error": "Invalid response_format"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files
        try:
            os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    args = get_args()
    
    # Validate required arguments
    if not os.path.exists(args.model_config):
        print(f"Error: Model config file {args.model_config} not found")
        exit(1)
    
    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Model checkpoint file {args.model_checkpoint} not found")
        exit(1)
    
    # Store args in app config
    app.config['args'] = args
    
    print(f"Starting API server on port {args.port}...")
    print(f"Model config: {args.model_config}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Device: {args.device}")
    
    app.run(host='0.0.0.0', port=args.port, debug=False) 