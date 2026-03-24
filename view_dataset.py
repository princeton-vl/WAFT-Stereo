import os
import argparse
import imageio.v2 as imageio
import cv2
import numpy as np
import torch
import copy

from bridgedepth.utils import visualization
from bridgedepth.dataloader.datasets import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tartanair", type=str, help="Dataset to visualize: fsd, tartanair, cres")
    args = parser.parse_args()
    if args.dataset.lower() == "fsd":
        d = FSD()
    elif args.dataset.lower() == "tartanair":
        d = TartanAir()
    elif args.dataset.lower() == "crestereo":
        d = CREStereo()
    elif args.dataset.lower() == "sceneflow":
        d = SceneFlowDatasets(dstype='frames_finalpass')
    elif args.dataset.lower() == "spring":
        d = Spring()
    elif args.dataset.lower() == "booster":
        d = Booster()
    elif args.dataset.lower() == "tartanground":
        d = TartanGround()
    elif args.dataset.lower() == "sintelstereo":
        d = SintelStereo()
    elif args.dataset.lower() == "fallingthings":
        d = FallingThings()
    elif args.dataset.lower() == "instereo2k":
        d = InStereo2K()
    elif args.dataset.lower() == "carlahighres":
        d = CarlaHighres()
    elif args.dataset.lower() == "vkitti2":
        d = VirtualKitti2()
    elif args.dataset.lower() == "unrealstereo4k":
        d = UnrealStereo4K()
    elif args.dataset.lower() == "wmgstereo":
        d = WMGStereo()
    elif args.dataset.lower() == "eth3d":
        d = ETH3D()
    elif args.dataset.lower() == "middlebury":
        d = Middlebury(split='F', image_set='training')
    elif args.dataset.lower() == "kitti":
        d = KITTI(image_set='2015', split='training')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(len(d))
    sample = d[888]
    print(sample['img1'].shape, sample['img2'].shape, sample['disp'].shape, sample['valid'].shape)
    path = 'vis/datasets/' + args.dataset.lower()
    os.makedirs(path, exist_ok=True)
    img1 = sample['img1'].cpu().numpy().transpose(1, 2, 0)
    img2 = sample['img2'].cpu().numpy().transpose(1, 2, 0)
    viz = visualization.Visualizer(img1)
    cv2.imwrite(os.path.join(path, 'img1.png'), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(path, 'img2.png'), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    disp = sample['disp'].cpu().numpy().squeeze()
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    print(disp.min(), disp.max(), np.median(disp), np.mean(disp))
    vis_disp = viz.draw_disparity(disp, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    vis_disp = np.concatenate([img1, vis_disp], axis=1)
    cv2.imwrite(os.path.join(path, 'disp_gt.png'), cv2.cvtColor(vis_disp, cv2.COLOR_RGB2BGR))

