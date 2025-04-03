import requests
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Completion API Client')
    parser.add_argument('--server_url', type=str, required=True, help='API server URL (e.g., http://localhost:4011)')
    
    # Add subparsers for different command types
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Folder processing command
    folder_parser = subparsers.add_parser('folder', help='Process all files in a folder')
    folder_parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder containing point cloud files')
    folder_parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder for completed point clouds')
    folder_parser.add_argument('--target_points', type=int, default=8192, help='Target number of points for sampling')
    folder_parser.add_argument('--sampling_method', choices=['fps', 'random', 'voxel'], default='fps', help='Sampling method')
    folder_parser.add_argument('--file_extension', type=str, default='.ply', help='File extension to process')
    
    # Single file processing command
    file_parser = subparsers.add_parser('file', help='Process a single file')
    file_parser.add_argument('--input_file', type=str, required=True, help='Path to input point cloud file')
    file_parser.add_argument('--output_file', type=str, required=True, help='Path to output completed point cloud file')
    file_parser.add_argument('--target_points', type=int, default=8192, help='Target number of points for sampling')
    file_parser.add_argument('--sampling_method', choices=['fps', 'random', 'voxel'], default='fps', help='Sampling method')
    
    args = parser.parse_args()
    
    # Check server health
    try:
        health_url = f"{args.server_url}/health"
        health_response = requests.get(health_url)
        health_response.raise_for_status()  # Raise exception for HTTP errors
        print("Server is healthy")
    except Exception as e:
        print(f"Server health check failed: {str(e)}")
        return
    
    if args.command == 'folder':
        process_folder(args)
    elif args.command == 'file':
        process_file(args)
    else:
        parser.print_help()

def process_folder(args):
    url = f"{args.server_url}/complete_folder"
    
    # Prepare request data
    data = {
        "input_folder": args.input_folder,
        "output_folder": args.output_folder,
        "target_points": args.target_points,
        "sampling_method": args.sampling_method,
        "file_extension": args.file_extension
    }
    
    print(f"Processing all {args.file_extension} files in '{args.input_folder}'...")
    
    try:
        # Send POST request with JSON data
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        print(f"Processing complete! Successful: {result['successful']}/{result['total_files']}")
        
        # Print detailed results if there were failures
        if result['successful'] < result['total_files']:
            print("\nFailed files:")
            for file_result in result['results']:
                if file_result['status'] == 'failed':
                    print(f"  - {file_result['file']}: {file_result.get('error', 'Unknown error')}")
                    
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            try:
                error_details = json.loads(e.response.text)
                print(f"Server error: {error_details.get('detail', e.response.text)}")
            except:
                print(f"Server response: {e.response.text}")

def process_file(args):
    url = f"{args.server_url}/complete_file"
    
    # Prepare request data
    data = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "target_points": args.target_points,
        "sampling_method": args.sampling_method
    }
    
    print(f"Processing file '{args.input_file}'...")
    
    try:
        # Send POST request with JSON data
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        print(f"Processing complete! Output saved to: {result['output_file']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            try:
                error_details = json.loads(e.response.text)
                print(f"Server error: {error_details.get('detail', e.response.text)}")
            except:
                print(f"Server response: {e.response.text}")

if __name__ == "__main__":
    main() 