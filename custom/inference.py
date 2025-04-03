##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
import open3d as o3d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose



def inference_single(model, pcd, args, config):
    # if root is not None:
    #     pc_file = os.path.join(root, pc_path)
    # else:
    #     pc_file = pc_path
    # read single point cloud
    # pc_ndarray = IO.get(pc_file).astype(np.float32)
    # transform it according to the model
    pc_ndarray =pcd
    
    # Check if pcd is an Open3D PointCloud object and convert it to numpy array if needed
    if isinstance(pc_ndarray, o3d.geometry.PointCloud):
        pc_ndarray = np.asarray(pc_ndarray.points)
        
    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    return dense_points

def Inference(pcd,args):
    # args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    pcd = inference_single(base_model, pcd, args, config)

    return pcd


def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        for pc_file in pc_file_list:
            inference_single(base_model, pc_file, args, config, root=args.pc_root)
    else:
        inference_single(base_model, args.pc, args, config)

if __name__ == '__main__':
    main()