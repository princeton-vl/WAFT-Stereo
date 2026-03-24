import os
import argparse
import imageio.v2 as imageio
import cv2
import numpy as np
import torch
import random
import copy
import torch.nn.functional as F

from algorithms.waft import WAFT
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils import visualization
from bridgedepth.dataloader.datasets import KITTI, SceneFlowDatasets, ETH3D, Middlebury, FSD
from bridgedepth.loss import build_criterion

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def macs_profiler(model):
    input = torch.randn(1, 3, 544, 960).cuda()
    sample = {
        "img1": input,
        "img2": input,
    }
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_flops=True) as prof:
                output = model(sample)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")
    print("Number of parameters: ", count_parameters(model) / 1e6, "M")

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(image, heatmap):
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    overlay = add_color_bar_to_image(overlay, color_bar, orientation='vertical')
    return overlay

def get_heatmap(info):
    weight = info[:, :2].softmax(dim=1)
    heatmap = weight[:, 0]
    h, w = heatmap.shape[-2:]
    return heatmap.view(h, w)

def demo_stereo(sample, model, output_dir, factor_list=[1.0]):
    os.makedirs(output_dir, exist_ok=True)
    for k in sample.keys():
        sample[k] = sample[k].cuda()

    sample['disp'] = torch.nan_to_num(sample['disp'], nan=0.0, posinf=0.0, neginf=0.0)
    input_sample = {
        "img1": sample["img1"].unsqueeze(0),
        "img2": sample["img2"].unsqueeze(0),
        "disp": sample["disp"].unsqueeze(0),
        "valid": sample["valid"].unsqueeze(0),
    }
    print(f"Input Resolution: {input_sample['img1'].shape}")
    print(f"Disparity Range: {sample['disp'].min().item()} - {sample['disp'].max().item()}")
    with torch.no_grad():
        # results_dict = model.inference(input_sample, size=model.cfg.DATASETS.CROP_SIZE, factor=factor_list)
        # results_dict = model.heirarchical_inference(input_sample, size=model.cfg.DATASETS.CROP_SIZE, factor_list=factor_list)
        results_dict = model(input_sample)
        
    H, W = sample['img1'].shape[-2:]
    img1 = sample['img1'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img2 = sample['img2'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'img1.png'), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'img2.png'), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    viz = visualization.Visualizer(img1)
    disp = results_dict['disp_pred'].cpu().numpy().reshape(H, W)
    vis_disp = viz.draw_disparity(disp, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    vis_disp = np.concatenate([img1, vis_disp], axis=1)
    cv2.imwrite(os.path.join(output_dir, 'disp_pred.png'), cv2.cvtColor(vis_disp, cv2.COLOR_RGB2BGR))

    disp_gt = sample['disp'].cpu().numpy().reshape(H, W)
    valid_gt = sample['valid'].cpu().numpy().reshape(H, W).astype(bool)
    print(f"Number of valid pixels in gt: {valid_gt.sum()}")

    disp_gt[~valid_gt] = 0
    vis_disp_gt = viz.draw_disparity(disp_gt, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    cv2.imwrite(os.path.join(output_dir, 'disp_gt.png'), cv2.cvtColor(vis_disp_gt, cv2.COLOR_RGB2BGR))

    disp_error = np.abs(disp - disp_gt)
    disp_error[~valid_gt] = 0
    vis_disp_error = viz.draw_disparity(disp_error, colormap=cv2.COLORMAP_JET, enhance=False).get_image()
    cv2.imwrite(os.path.join(output_dir,'disp_error.png'), cv2.cvtColor(vis_disp_error, cv2.COLOR_RGB2BGR))
    print(f"Mean disparity error (valid): {disp_error[valid_gt].mean()}, Max disparity error: {disp_error.max()}")

    if 'init' in results_dict.keys():
        bins_prob_init = results_dict['init'].softmax(dim=1)
        bins_idx = torch.linspace(0, model.max_disp, model.n_bins, device=bins_prob_init.device, dtype=bins_prob_init.dtype).view(1, model.n_bins, 1, 1)
        disp_bins_init = torch.sum(bins_prob_init * bins_idx, dim=1)[0].cpu().numpy() / factor_list[-1]
        vis_disp_bins_init = viz.draw_disparity(disp_bins_init, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
        cv2.imwrite(os.path.join(output_dir, 'init_disp.png'), cv2.cvtColor(vis_disp_bins_init, cv2.COLOR_RGB2BGR))
        disp_error = np.abs(disp_bins_init - disp_gt)
        disp_error[~valid_gt] = 0
        vis_disp_error = viz.draw_disparity(disp_error, colormap=cv2.COLORMAP_JET, enhance=False).get_image()
        cv2.imwrite(os.path.join(output_dir,f"init_error.png"), cv2.cvtColor(vis_disp_error, cv2.COLOR_RGB2BGR))
        epe_bins_init = disp_error[valid_gt].mean()
        bp1_bins_init = (disp_error[valid_gt] > 1.0).sum() / valid_gt.sum()
        bp2_bins_init = (disp_error[valid_gt] > 2.0).sum() / valid_gt.sum()
        print(f"Init disparity EPE: {epe_bins_init}, BP1: {bp1_bins_init}, BP2: {bp2_bins_init}")

    for i in range(model.iters):
        heatmap = get_heatmap(results_dict['delta_info_preds'][i]).cpu().numpy()
        vis_uncertainty = vis_heatmap(img1, heatmap)
        cv2.imwrite(os.path.join(output_dir, f"uncertainty_step{i}.png"), vis_uncertainty)
        disp_i = results_dict['delta_disp_preds'][i][0, 0].cpu().numpy()
        vis_disp_i = viz.draw_disparity(disp_i, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
        cv2.imwrite(os.path.join(output_dir,f"disp_step{i}.png"), cv2.cvtColor(vis_disp_i, cv2.COLOR_RGB2BGR))
        disp_error = np.abs(disp_i - disp_gt)
        disp_error[~valid_gt] = 0
        vis_disp_error = viz.draw_disparity(disp_error, colormap=cv2.COLORMAP_JET, enhance=False).get_image()
        cv2.imwrite(os.path.join(output_dir,f"disp_error_step{i}.png"), cv2.cvtColor(vis_disp_error, cv2.COLOR_RGB2BGR))
        epe_lap_i = disp_error[valid_gt].mean()
        bp1_lap_i = (disp_error[valid_gt] > 1.0).sum() / valid_gt.sum()
        bp2_lap_i = (disp_error[valid_gt] > 2.0).sum() / valid_gt.sum()
        print(f"Step{i} disparity EPE: {epe_lap_i}, BP1: {bp1_lap_i}, BP2: {bp2_lap_i}")


def vis_middlebury(model):
    dataset = Middlebury(split='H', image_set='training')
    for idx in [1, 5, 6, 8, 9]:
        sample = dataset[idx]
        print(f'Visualizing sample {idx}...')
        demo_stereo(sample, model, output_dir=f'vis/middlebury/{idx}', factor_list=[1.0])

def vis_ETH3D(model):
    dataset = ETH3D(split='training')
    idx = 10
    sample = dataset[idx]
    demo_stereo(sample, model, output_dir=f'vis/ETH3D/{idx}')

def vis_kitti(model):
    dataset = KITTI(image_set='2015', split='training')
    for idx in [0, 100]:
        sample = dataset[idx]
        demo_stereo(sample, model, output_dir=f'vis/kitti/{idx}')

def vis_things(model):
    dataset = SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    sample = dataset[0]
    demo_stereo(sample, model, output_dir='vis/things')

def vis_fsd(model):
    dataset = FSD(size=10000)
    sample = dataset[1000]
    demo_stereo(sample, model, output_dir='vis/fsd')

def vis_tartanground(model, idxes=[]):
    from bridgedepth.dataloader.datasets import TartanGround
    dataset = TartanGround()
    for idx in idxes:
        print(f'Visualizing sample {idx}...')
        sample = dataset[idx]
        demo_stereo(sample, model, output_dir=f"vis/tartanground/{idx}")

def vis_booster(model):
    from bridgedepth.dataloader.datasets import Booster
    dataset = Booster(resolution='Q')
    print(len(dataset))
    idx_list = random.sample(range(len(dataset)), 5)
    for idx in idx_list:
        print(f'Visualizing sample {idx}...')
        sample = dataset[idx]
        demo_stereo(sample, model, output_dir=f'vis/booster/{idx}', factor_list=[1.0])

def vis_vkitti2(model):
    from bridgedepth.dataloader.datasets import VirtualKitti2
    dataset = VirtualKitti2()
    sample = dataset[0]
    demo_stereo(sample, model, output_dir='vis/vkitti2')

def vis_sintelstereo(model):
    from bridgedepth.dataloader.datasets import SintelStereo
    dataset = SintelStereo()
    sample = dataset[123]
    demo_stereo(sample, model, output_dir='vis/sintelstereo')

def vis_fallingthings(model):
    from bridgedepth.dataloader.datasets import FallingThings
    dataset = FallingThings()
    sample = dataset[0]
    demo_stereo(sample, model, output_dir='vis/fallingthings')

def vis_instereo2k(model):
    from bridgedepth.dataloader.datasets import InStereo2K
    dataset = InStereo2K()
    for idx in [0, 100, 200, 300, 400, 500]:
        print(f'Visualizing sample {idx}...')
        sample = dataset[idx]
        demo_stereo(sample, model, output_dir=f'vis/instereo2k/{idx}', factor_list=[1.0])

def vis_carlahighres(model):
    from bridgedepth.dataloader.datasets import CarlaHighres
    dataset = CarlaHighres()
    sample = dataset[89]
    demo_stereo(sample, model, output_dir='vis/carlahighres')

def vis_unrealstereo4k(model):
    from bridgedepth.dataloader.datasets import UnrealStereo4K
    dataset = UnrealStereo4K()
    sample = dataset[0]
    demo_stereo(sample, model, output_dir='vis/unrealstereo4k')

def vis_wmgstereo(model):
    from bridgedepth.dataloader.datasets import WMGStereo
    dataset = WMGStereo()
    sample = dataset[20000]
    demo_stereo(sample, model, output_dir='vis/wmgstereo')

def vis_spring(model):
    from bridgedepth.dataloader.datasets import Spring
    dataset = Spring()
    sample = dataset[4444]
    demo_stereo(sample, model, output_dir='vis/spring', factor_list=[0.5, 1.0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()
    
    cfg = setup(args)
    model = WAFT(cfg)
    model.eval()
    model = model.to(torch.device("cuda"))
    test_model = copy.deepcopy(model)
    macs_profiler(test_model)

    if args.ckpt is not None:
        print('Load checkpoint: %s' % args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(weights, strict=False)

    # vis_middlebury(model)
    # vis_ETH3D(model)
    # vis_kitti(model)
    # vis_things(model)
    # vis_fsd(model)
    # vis_booster(model)
    # vis_vkitti2(model)
    # vis_sintelstereo(model)
    # vis_fallingthings(model)
    vis_instereo2k(model)
    # vis_carlahighres(model)
    # vis_unrealstereo4k(model)
    # vis_wmgstereo(model)
    # import random
    # idxes = [random.randint(0, 400000) for _ in range(20)]
    # vis_tartanground(model, idxes=idxes)
    # vis_spring(model)