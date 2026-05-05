# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
from peft import PeftModel
from algorithms.waft import WAFT
from bridgedepth.utils import visualization
from visualize import vis_heatmap, get_heatmap

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def depth2xyzmap(depth: np.ndarray, K, uvs: np.array=None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us] 
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N, 3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud

def setup(args):
    """
    Create config and perform basic setups.
    """
    from bridgedepth.config import get_cfg
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

if __name__=="__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=f"{code_dir}/assets/Keyboard", type=str, help='dir')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--get_pc', type=int, default=0, help='save point cloud output')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    cfg = setup(args)
    model = WAFT(cfg)
    model.eval()
    model = model.to(torch.device("cuda"))
    print('Load checkpoint: %s' % args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)
    for name, module in model.named_modules():
        if isinstance(module, PeftModel):
            print(f"{name} is a PeftModel with {count_parameters(module)} trainable parameters.")
            module.merge_and_unload()

    img0 = imageio.imread(f"{args.dir}/left.png")
    img1 = imageio.imread(f"{args.dir}/right.png")
    img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
    img1 = cv2.resize(img1, fx=args.scale, fy=args.scale, dsize=None)
    H,W = img0.shape[:2]
    print(img0.shape, img1.shape)
    input_sample = {
        "img1": torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2),
        "img2": torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    }
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            results_dict = model(input_sample)

    viz = visualization.Visualizer(img0)
    disp = results_dict['disp_pred'].cpu().numpy().reshape(H, W)
    vis_disp = viz.draw_disparity(disp, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    vis_disp = np.concatenate([img0, vis_disp], axis=1)
    imageio.imwrite(f'{args.dir}/vis_disp.png', vis_disp)

    heatmap = get_heatmap(results_dict['delta_info_preds'][-1]).cpu().numpy()
    vis_uncertainty = vis_heatmap(img0, heatmap)
    cv2.imwrite(f"{args.dir}/uncertainty.png", vis_uncertainty)

    if args.remove_invisible:
        yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = us_right<0
        disp[invalid] = np.inf

    if args.get_pc:
        intrinsic_file = f"{args.dir}/K.txt"
        with open(intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
        
        K[:2] = K[:2] * args.scale
        depth = K[0,0]*baseline/disp
        np.save(f'{args.dir}/depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0.reshape(-1,3))
        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(f'{args.dir}/cloud.ply', pcd)
        print(f"PCL saved to {args.dir}")

        if args.denoise_cloud:
            logging.info("[Optional step] denoise point cloud...")
            cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(f'{args.dir}/cloud_denoise.ply', inlier_cloud)
            pcd = inlier_cloud
