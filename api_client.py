import requests
import argparse
import os
import base64

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Completion API Client')
    parser.add_argument('--server_url', type=str, required=True, help='API server URL (e.g., http://localhost:5000)')
    parser.add_argument('--point_cloud_file', type=str, required=True, help='Path to point cloud file (.ply)')
    parser.add_argument('--target_points', type=int, default=8192, help='Target number of points for sampling')
    parser.add_argument('--sampling_method', choices=['fps', 'random', 'voxel'], default='fps', help='Sampling method')
    parser.add_argument('--output_file', type=str, help='Output file path for completed point cloud')
    parser.add_argument('--format', choices=['file', 'base64'], default='file', help='Response format')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.point_cloud_file):
        print(f"Error: Point cloud file {args.point_cloud_file} not found")
        return
    
    # Check server health
    try:
        health_url = f"{args.server_url}/health"
        health_response = requests.get(health_url)
        health_response.raise_for_status()  # Raise exception for HTTP errors
        print("Server is healthy")
    except Exception as e:
        print(f"Server health check failed: {str(e)}")
        return
    
    # Prepare the API request
    url = f"{args.server_url}/complete"
    
    # Prepare form data and files
    data = {
        'target_points': str(args.target_points),
        'sampling_method': args.sampling_method,
        'response_format': args.format
    }
    
    files = {
        'file': (os.path.basename(args.point_cloud_file), open(args.point_cloud_file, 'rb'))
    }
    
    print(f"Sending point cloud to server for completion...")
    
    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        if args.format == 'file':
            # If output file wasn't specified, use input filename with _completed suffix
            if not args.output_file:
                base_name = os.path.splitext(args.point_cloud_file)[0]
                args.output_file = f"{base_name}_completed.ply"
                
            # Save the received file
            with open(args.output_file, 'wb') as f:
                f.write(response.content)
            print(f"Completed point cloud saved to {args.output_file}")
            
        elif args.format == 'base64':
            # Process base64 response
            json_response = response.json()
            if json_response.get('completion_successful'):
                # If output file is specified, save the base64 content
                if args.output_file:
                    with open(args.output_file, 'wb') as f:
                        decoded = base64.b64decode(json_response['point_cloud_base64'])
                        f.write(decoded)
                    print(f"Completed point cloud saved to {args.output_file}")
                else:
                    print("Received base64 encoded point cloud")
            else:
                print("Error in completion process")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")

if __name__ == "__main__":
    main() 