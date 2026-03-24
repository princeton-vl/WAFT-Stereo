import argparse
import multiprocessing as mp
import os
import time
import torch
import cv2
import copy
import numpy as np

from bridgedepth.config import get_cfg
from bridgedepth.utils.logger import setup_logger
from bridgedepth.dataloader import datasets
from bridgedepth.utils import frame_utils, visualization
from algorithms.waft import WAFT


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--ckpt", default=None, help='path to the checkpoint file or model name when eval_only is True')
    parser.add_argument("--dataset", help="Dataset name to generate prediction results")
    parser.add_argument(
        "--output",
        help="A file or directory to save prediction results.",
    )
    parser.add_argument(
        "--show-attr",
        default="disparity",
        help="The attribute to visualize.",
    )
    return parser


@torch.no_grad()
def run_on_dataset(dataset, model, output, find_output_path=None, show_attr="disparity"):
    model.eval()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        rgb = sample["img1"].permute(1, 2, 0).numpy()
        viz = visualization.Visualizer(rgb)

        sample["img1"] = sample["img1"][None]
        sample["img2"] = sample["img2"][None]
        result_dict = model(sample)

        if show_attr == "error":
            valid = sample["valid"]
            disp_gt = sample["disp"]
            disp_pred = result_dict["disp"][0].to(disp_gt.device)
            error = torch.abs(disp_pred - disp_gt).abs()
            # valid mask
            valid = valid & (disp_gt > 0) & (disp_gt < cfg.TEST.EVAL_MAX_DISP[0])
            error[~valid] = 0
            visualized_output = viz.draw_error_map(error)
        elif show_attr == "disparity":
            disp_pred = result_dict["disp_pred"][0].cpu()
            visualized_output = viz.draw_disparity(disp_pred, colormap="kitti")
        else:
            raise ValueError(f"not supported visualization attribute {show_attr}")

        file_path = dataset.image_list[idx][0]
        if output:
            assert find_output_path is not None
            output_path = os.path.join(output, find_output_path(file_path))
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)
            visualized_output.save(output_path)
        else:
            cv2.namedWindow(f"{show_attr}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{show_attr}", visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit


@torch.no_grad()
def create_kitti_submission(model, image_set, output):
    training_mode = model.training
    model.eval()
    test_dataset = datasets.KITTI(split='testing', image_set=image_set)
    
    total_compute_time = 0.0

    output_path = os.path.join(output, f'{image_set}_submission')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # warm up
    sample = test_dataset[0]
    sample = {"img1": sample['img1'][None].cuda(), "img2": sample['img2'][None].cuda()}
    model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for test_id in range(len(test_dataset)):
        sample = test_dataset[test_id]
        frame_id = sample['meta'].split('/')[-1]
        sample = {"img1": sample['img1'][None].cuda(), "img2": sample['img2'][None].cuda()}

        start_compute_time = time.perf_counter()
        results_dict = model.inference(sample, size=model.cfg.DATASETS.CROP_SIZE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_compute_time += time.perf_counter() - start_compute_time
        disp = results_dict['disp_pred'][0].cpu().numpy()
        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeDispKITTI(output_filename, disp)

    model.train(training_mode)
    print('The average runtime on Kitti test images is (you will need this for the submission): '
          f'{total_compute_time / len(test_dataset):.4f} s/iter')


@torch.no_grad()
def create_eth3d_submission(model, output_path):
    training_mode = model.training
    model.eval()

    test_dataset = datasets.ETH3D(split='test')
    test_dataset.is_test = True
    training_dataset = datasets.ETH3D(split='training')
    training_dataset.is_test = True
    dataset = test_dataset + training_dataset

    # warm up
    sample = dataset[0]
    sample = {"img1": sample['img1'][None].cuda(), "img2": sample['img2'][None].cuda()}
    model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for test_id in range(len(dataset)):
        sample = dataset[test_id]
        imageL_file = sample['meta']
        sample = {"img1": sample['img1'][None].cuda(), "img2": sample['img2'][None].cuda()}

        start_compute_time = time.perf_counter()
        results_dict = model.inference(sample, size=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - start_compute_time

        disp = results_dict['disp_pred'][0].cpu().numpy()
        # disp[disp > 64] = 64

        names = imageL_file.split("/")
        save_sub_path = os.path.join(output_path, "low_res_two_view")
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)

        disp_path = os.path.join(save_sub_path, names[-2] + '.pfm')
        frame_utils.writePFM(disp_path, disp)
        
        txt_path = os.path.join(save_sub_path, names[-2] + '.txt')
        with open(txt_path, 'wb') as time_file:
            time_file.write(bytes('runtime ' + str(compute_time), 'UTF-8'))

    model.train(training_mode)


@torch.no_grad()
def create_middlebury_submission(model, output_path, split='F', method_name='WAFTStereo'):
    training_mode = model.training
    model.eval()
    test_dataset = datasets.Middlebury(split=split, image_set='test')
    test_dataset.is_test = True
    training_dataset = datasets.Middlebury(split=split, image_set='training')
    training_dataset.is_test = True
    dataset = test_dataset + training_dataset
    torch.backends.cudnn.benchmark = True

    # warm up
    sample = dataset[0]
    sample = {"img1": sample['img1'][None].cuda(), "img2": sample['img2'][None].cuda()}
    model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for test_id in range(len(dataset)):
        sample = dataset[test_id]
        imageL_file = sample['meta']
        sample = {"img1": sample['img1'][None].cuda(), "img2": sample['img2'][None].cuda()}

        start_compute_time = time.perf_counter()
        results_dict = model.heirarchical_inference(sample, size=None, factor_list=[0.5, 1.0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - start_compute_time

        disp = results_dict['disp_pred'][0].cpu().numpy()
        # disp[disp > 400] = 400

        names = imageL_file.split("/")
        save_sub_path = os.path.join(output_path, names[-3], names[-2])
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)

        disp_path = os.path.join(save_sub_path, 'disp0' + method_name + '.pfm')
        frame_utils.writePFM(disp_path, disp)

        txt_path = os.path.join(save_sub_path, 'time' + method_name + '.txt')
        with open(txt_path, 'wb') as time_file:
            time_file.write(bytes(str(compute_time), 'UTF-8'))

    model.train(training_mode)

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="waft")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    model = WAFT(cfg)
    model = model.to(torch.device("cuda")).eval()
    print('Load checkpoint: %s' % args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)

    output = args.output
    if 'kitti' in args.dataset:
        output = args.output if args.output else '.'
        create_kitti_submission(model, args.dataset.split('_')[-1], output)
    elif args.dataset == 'eth3d':
        output = args.output if args.output else '.'
        create_eth3d_submission(model, output)
    elif args.dataset.startswith("middlebury_"):
        output = args.output if args.output else '.'
        create_middlebury_submission(model, output, split=args.dataset.replace('middlebury_', ''))
    else:
        raise ValueError(f"Not supported dataset {args.dataset} for inference")